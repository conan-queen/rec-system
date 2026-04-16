"""
api/server.py（升级版）
加入 Redis 缓存 + 推荐日志记录

新增接口：
  POST /rate            → 用户提交评分，自动清除该用户缓存
  GET  /cache/stats     → 缓存命中率统计
  DELETE /cache         → 手动清除所有缓存（模型重训后调用）
"""
from __future__ import annotations

import os
import sys
import pickle
import time
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from models.collaborative import SVDRecommender
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender
from cache.redis_cache import get_cache


app = FastAPI(
    title="电影推荐系统 API（生产版）",
    description="MySQL 数据 + Redis 缓存 + 推荐日志",
    version="2.0.0"
)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

_models = {}


def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    required = ["cf_model.pkl", "cb_model.pkl", "processor.pkl",
                "movies.pkl", "matrix.pkl"]
    for fname in required:
        if not os.path.exists(os.path.join(models_dir, fname)):
            print(f"[API] 模型文件不存在: {fname}，请先运行 train.py")
            return False

    _models["cf"] = SVDRecommender.load(os.path.join(models_dir, "cf_model.pkl"))
    _models["cb"] = ContentBasedRecommender.load(
        os.path.join(models_dir, "cb_model.pkl"))
    _models["hybrid"] = HybridRecommender(
        cf_model=_models["cf"], cb_model=_models["cb"]
    )
    with open(os.path.join(models_dir, "processor.pkl"), "rb") as f:
        _models["processor"] = pickle.load(f)
    with open(os.path.join(models_dir, "movies.pkl"), "rb") as f:
        _models["movies"] = pickle.load(f)
    with open(os.path.join(models_dir, "matrix.pkl"), "rb") as f:
        _models["matrix"] = pickle.load(f)

    print("[API] 所有模型加载成功")
    return True


@app.on_event("startup")
async def startup_event():
    load_models()
    get_cache()   # Redis 连接（失败自动降级，不影响服务）


# ── 请求 / 响应数据模型 ────────────────────────────────────────

class MovieItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float
    source: str


class RecommendResponse(BaseModel):
    user_id: int
    top_k: int
    recommendations: List[MovieItem]
    latency_ms: float
    cache_hit: bool        # 新增：是否命中缓存，方便监控缓存效果


class SimilarResponse(BaseModel):
    movie_id: int
    title: str
    similar_movies: List[MovieItem]
    latency_ms: float
    cache_hit: bool


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float


class RatingResponse(BaseModel):
    success: bool
    message: str


class CacheStatsResponse(BaseModel):
    available: bool
    cached_keys: Optional[int] = None
    hits: Optional[int] = None
    misses: Optional[int] = None


# ── 接口实现 ───────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    cache_stats = get_cache().cache_stats()
    return {
        "status":          "ok" if _models else "degraded",
        "models_loaded":   bool(_models),
        "redis_available": cache_stats.get("available", False),
        "n_users":         _models["processor"].n_users if _models else None,
        "n_items":         _models["processor"].n_items if _models else None,
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    mode: str = Query(default="hybrid", pattern="^(cf|cb|hybrid)$")
):
    """
    个性化推荐（带 Redis 缓存）

    流程：查缓存 → 未命中则推理 → 写缓存 → 返回结果
    """
    if not _models:
        raise HTTPException(status_code=503, detail="模型未加载")

    processor = _models["processor"]
    user_idx = processor.user2idx.get(user_id)
    if user_idx is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    cache = get_cache()
    t0 = time.time()

    # ── 1. 查缓存 ─────────────────────────────────────────────
    cached = cache.get_recommendations(user_id, top_k, mode)
    if cached is not None:
        return RecommendResponse(
            user_id=user_id,
            top_k=top_k,
            recommendations=[MovieItem(**r) for r in cached],
            latency_ms=round((time.time() - t0) * 1000, 2),
            cache_hit=True
        )

    # ── 2. 缓存未命中，模型推理 ───────────────────────────────
    matrix = _models["matrix"]
    movies = _models["movies"]

    try:
        if mode == "cf":
            raw = _models["cf"].recommend(user_idx, matrix, top_k=top_k)
            recs_data = [{"item_idx": r[0], "score": r[1], "source": "cf"}
                         for r in raw]
        elif mode == "cb":
            raw = _models["cb"].recommend_for_user(
                user_id=user_id, interaction_matrix=matrix,
                user_idx=user_idx, item2idx=processor.item2idx,
                idx2item=processor.idx2item, top_k=top_k
            )
            recs_data = []
            for movie_id, score in raw:
                idx = processor.item2idx.get(movie_id)
                if idx is not None:
                    recs_data.append({"item_idx": idx,
                                      "score": score, "source": "cb"})
        else:
            recs_data = _models["hybrid"].recommend(
                user_idx=user_idx, interaction_matrix=matrix,
                item2idx=processor.item2idx,
                idx2item=processor.idx2item, top_k=top_k
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")

    # ── 3. 组装响应 ───────────────────────────────────────────
    recommendations = []
    cache_data = []
    for r in recs_data:
        movie_id = processor.idx2item.get(r["item_idx"])
        if movie_id is None:
            continue
        movie_row = movies[movies.movie_id == movie_id]
        if movie_row.empty:
            continue
        item = {
            "movie_id": int(movie_id),
            "title":    movie_row.title.values[0],
            "genres":   movie_row.genres.values[0],
            "score":    float(r["score"]),
            "source":   r.get("source", mode)
        }
        recommendations.append(MovieItem(**item))
        cache_data.append(item)

    # ── 4. 写缓存 ─────────────────────────────────────────────
    n_interactions = matrix[user_idx].nnz
    cache.set_recommendations(user_id, top_k, cache_data, mode, n_interactions)

    return RecommendResponse(
        user_id=user_id,
        top_k=top_k,
        recommendations=recommendations,
        latency_ms=round((time.time() - t0) * 1000, 2),
        cache_hit=False
    )


@app.get("/similar/{movie_id}", response_model=SimilarResponse)
async def similar_movies(
    movie_id: int,
    top_k: int = Query(default=10, ge=1, le=20)
):
    """相似电影推荐（带缓存，24小时过期）"""
    if not _models:
        raise HTTPException(status_code=503, detail="模型未加载")

    movies = _models["movies"]
    movie_row = movies[movies.movie_id == movie_id]
    if movie_row.empty:
        raise HTTPException(status_code=404, detail=f"电影 {movie_id} 不存在")

    cache = get_cache()
    t0 = time.time()

    cached = cache.get_similar(movie_id, top_k)
    if cached is not None:
        return SimilarResponse(
            movie_id=movie_id,
            title=movie_row.title.values[0],
            similar_movies=[MovieItem(**r) for r in cached],
            latency_ms=round((time.time() - t0) * 1000, 2),
            cache_hit=True
        )

    similar = _models["cb"].get_similar_movies(movie_id=movie_id, top_k=top_k)
    similar_items = []
    cache_data = []
    for mid, score in similar:
        row = movies[movies.movie_id == mid]
        if row.empty:
            continue
        item = {
            "movie_id": int(mid),
            "title":    row.title.values[0],
            "genres":   row.genres.values[0],
            "score":    round(score, 4),
            "source":   "cb"
        }
        similar_items.append(MovieItem(**item))
        cache_data.append(item)

    cache.set_similar(movie_id, top_k, cache_data)

    return SimilarResponse(
        movie_id=movie_id,
        title=movie_row.title.values[0],
        similar_movies=similar_items,
        latency_ms=round((time.time() - t0) * 1000, 2),
        cache_hit=False
    )


@app.post("/rate", response_model=RatingResponse)
async def submit_rating(req: RatingRequest):
    """
    用户提交评分

    提交后立即清除该用户的推荐缓存，
    下次推荐请求会重新计算（但模型本身等定时重训更新）
    """
    if not 1.0 <= req.rating <= 5.0:
        raise HTTPException(status_code=400,
                            detail="评分须在 1.0 ~ 5.0 之间")

    # 生产环境：在此写入 MySQL
    # db_loader.save_ratings([{"user_id": req.user_id,
    #                           "movie_id": req.movie_id,
    #                           "rating": req.rating}])

    deleted = get_cache().invalidate_user(req.user_id)
    return RatingResponse(
        success=True,
        message=f"评分已提交，已清除 {deleted} 个旧缓存"
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """查看缓存命中率（运维监控用）"""
    return CacheStatsResponse(**get_cache().cache_stats())


@app.delete("/cache")
async def clear_all_cache():
    """
    清除全部推荐缓存

    使用场景：python train.py 重训完成后调用此接口，
    让所有用户下次请求时拿到新模型的推荐结果
    """
    count = get_cache().invalidate_all()
    return {"message": f"已清除 {count} 个缓存 key"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
