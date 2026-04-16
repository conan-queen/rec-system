"""
api/server.py v3.0
新增：电影搜索、排行榜、评论、海报、用户登录
"""
from __future__ import annotations

import os
import sys
import pickle
import time
import math
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.collaborative import SVDRecommender
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender
from cache.redis_cache import get_cache
from ab_test.ab_manager import ABManager

_ab = ABManager()

app = FastAPI(title="CineAI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

_models = {}

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def get_db():
    import pymysql
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "railway"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )


def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    required = ["cf_model.pkl", "cb_model.pkl", "processor.pkl", "movies.pkl", "matrix.pkl"]
    for fname in required:
        if not os.path.exists(os.path.join(models_dir, fname)):
            print(f"[API] 模型文件不存在: {fname}")
            return False
    _models["cf"] = SVDRecommender.load(os.path.join(models_dir, "cf_model.pkl"))
    _models["cb"] = ContentBasedRecommender.load(os.path.join(models_dir, "cb_model.pkl"))
    _models["hybrid"] = HybridRecommender(cf_model=_models["cf"], cb_model=_models["cb"])
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
    get_cache()


# ── 数据模型 ──────────────────────────────────

class MovieItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float
    source: str
    poster_path: Optional[str] = None
    overview: Optional[str] = None

class RecommendResponse(BaseModel):
    user_id: int
    top_k: int
    recommendations: List[MovieItem]
    latency_ms: float
    cache_hit: bool

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

class CommentRequest(BaseModel):
    user_id: int
    movie_id: int
    content: str

class CommentResponse(BaseModel):
    id: int
    user_id: int
    username: str
    movie_id: int
    content: str
    created_at: str

class MovieDetail(BaseModel):
    movie_id: int
    title: str
    genres: str
    year: Optional[int] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    avg_rating: Optional[float] = None
    rating_count: int = 0


# ── 辅助函数 ──────────────────────────────────

def get_poster_url(poster_path):
    if poster_path:
        return TMDB_IMAGE_BASE + poster_path
    return None

def safe_score(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return round(float(val), 4)

def enrich_with_poster(movie_id, movies_df):
    """从 DataFrame 获取海报路径"""
    row = movies_df[movies_df.movie_id == movie_id]
    if row.empty:
        return None, None
    poster = row.poster_path.values[0] if "poster_path" in row.columns else None
    overview = row.overview.values[0] if "overview" in row.columns else None
    return poster, overview


# ── 接口 ──────────────────────────────────────

@app.get("/health")
async def health_check():
    cache_stats = get_cache().cache_stats()
    return {
        "status": "ok" if _models else "degraded",
        "models_loaded": bool(_models),
        "redis_available": cache_stats.get("available", False),
        "n_users": _models["processor"].n_users if _models else None,
        "n_items": _models["processor"].n_items if _models else None,
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    mode: str = Query(default="hybrid", pattern="^(cf|cb|hybrid)$")
):
    if not _models:
        raise HTTPException(status_code=503, detail="模型未加载")

    processor = _models["processor"]
    user_idx = processor.user2idx.get(user_id)
    if user_idx is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    cache = get_cache()
    t0 = time.time()

    cached = cache.get_recommendations(user_id, top_k, mode)
    if cached is not None:
        return RecommendResponse(
            user_id=user_id, top_k=top_k,
            recommendations=[MovieItem(**r) for r in cached],
            latency_ms=round((time.time() - t0) * 1000, 2),
            cache_hit=True
        )

    matrix = _models["matrix"]
    movies = _models["movies"]

    try:
        if mode == "cf":
            raw = _models["cf"].recommend(user_idx, matrix, top_k=top_k)
            recs_data = [{"item_idx": r[0], "score": r[1], "source": "cf"} for r in raw]
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
                    recs_data.append({"item_idx": idx, "score": score, "source": "cb"})
        else:
            recs_data = _models["hybrid"].recommend(
                user_idx=user_idx, interaction_matrix=matrix,
                item2idx=processor.item2idx, idx2item=processor.idx2item, top_k=top_k
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")

    recommendations = []
    cache_data = []
    for r in recs_data:
        movie_id = processor.idx2item.get(int(r["item_idx"]))
        if movie_id is None:
            continue
        movie_row = movies[movies.movie_id == movie_id]
        if movie_row.empty:
            continue
        poster = movie_row.poster_path.values[0] if "poster_path" in movie_row.columns else None
        overview = movie_row.overview.values[0] if "overview" in movie_row.columns else None
        item = {
            "movie_id":   int(movie_id),
            "title":      movie_row.title.values[0],
            "genres":     movie_row.genres.values[0],
            "score":      safe_score(r["score"]),
            "source":     r.get("source", mode),
            "poster_path": get_poster_url(poster) if poster and str(poster) != "nan" else None,
            "overview":   overview if overview and str(overview) != "nan" else None,
        }
        recommendations.append(MovieItem(**item))
        cache_data.append(item)

    n_interactions = matrix[user_idx].nnz
    cache.set_recommendations(user_id, top_k, cache_data, mode, n_interactions)

    return RecommendResponse(
        user_id=user_id, top_k=top_k,
        recommendations=recommendations,
        latency_ms=round((time.time() - t0) * 1000, 2),
        cache_hit=False
    )


@app.get("/similar/{movie_id}", response_model=SimilarResponse)
async def similar_movies(
    movie_id: int,
    top_k: int = Query(default=10, ge=1, le=20)
):
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
            movie_id=movie_id, title=movie_row.title.values[0],
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
        poster = row.poster_path.values[0] if "poster_path" in row.columns else None
        overview = row.overview.values[0] if "overview" in row.columns else None
        item = {
            "movie_id":   int(mid),
            "title":      row.title.values[0],
            "genres":     row.genres.values[0],
            "score":      round(score, 4),
            "source":     "cb",
            "poster_path": get_poster_url(poster) if poster and str(poster) != "nan" else None,
            "overview":   overview if overview and str(overview) != "nan" else None,
        }
        similar_items.append(MovieItem(**item))
        cache_data.append(item)

    cache.set_similar(movie_id, top_k, cache_data)

    return SimilarResponse(
        movie_id=movie_id, title=movie_row.title.values[0],
        similar_movies=similar_items,
        latency_ms=round((time.time() - t0) * 1000, 2),
        cache_hit=False
    )


@app.get("/movies/search")
async def search_movies(q: str = Query(..., min_length=1)):
    """搜索电影"""
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT m.movie_id, m.title, m.genres, m.year,
                       m.poster_path, m.overview,
                       ROUND(AVG(r.rating), 1) as avg_rating,
                       COUNT(r.id) as rating_count
                FROM movies m
                LEFT JOIN ratings r ON m.movie_id = r.movie_id
                WHERE m.title LIKE %s
                GROUP BY m.movie_id
                LIMIT 20
            """, (f"%{q}%",))
            results = cursor.fetchall()
        conn.close()

        movies = []
        for row in results:
            movies.append({
                "movie_id":    row["movie_id"],
                "title":       row["title"],
                "genres":      row["genres"],
                "year":        row["year"],
                "poster_path": get_poster_url(row["poster_path"]) if row["poster_path"] else None,
                "overview":    row["overview"],
                "avg_rating":  float(row["avg_rating"]) if row["avg_rating"] else None,
                "rating_count": row["rating_count"],
            })
        return {"results": movies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/top")
async def top_movies(limit: int = Query(default=20, le=50)):
    """热门排行榜"""
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT m.movie_id, m.title, m.genres, m.year,
                       m.poster_path, m.overview,
                       ROUND(AVG(r.rating), 1) as avg_rating,
                       COUNT(r.id) as rating_count
                FROM movies m
                JOIN ratings r ON m.movie_id = r.movie_id
                GROUP BY m.movie_id
                HAVING rating_count >= 5
                ORDER BY avg_rating DESC, rating_count DESC
                LIMIT %s
            """, (limit,))
            results = cursor.fetchall()
        conn.close()

        movies = []
        for row in results:
            movies.append({
                "movie_id":    row["movie_id"],
                "title":       row["title"],
                "genres":      row["genres"],
                "year":        row["year"],
                "poster_path": get_poster_url(row["poster_path"]) if row["poster_path"] else None,
                "overview":    row["overview"],
                "avg_rating":  float(row["avg_rating"]) if row["avg_rating"] else None,
                "rating_count": row["rating_count"],
            })
        return {"movies": movies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/{movie_id}")
async def get_movie(movie_id: int):
    """电影详情"""
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT m.movie_id, m.title, m.genres, m.year,
                       m.poster_path, m.overview,
                       ROUND(AVG(r.rating), 1) as avg_rating,
                       COUNT(r.id) as rating_count
                FROM movies m
                LEFT JOIN ratings r ON m.movie_id = r.movie_id
                WHERE m.movie_id = %s
                GROUP BY m.movie_id
            """, (movie_id,))
            row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="电影不存在")

        return {
            "movie_id":    row["movie_id"],
            "title":       row["title"],
            "genres":      row["genres"],
            "year":        row["year"],
            "poster_path": get_poster_url(row["poster_path"]) if row["poster_path"] else None,
            "overview":    row["overview"],
            "avg_rating":  float(row["avg_rating"]) if row["avg_rating"] else None,
            "rating_count": row["rating_count"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/{movie_id}/comments")
async def get_comments(movie_id: int):
    """获取电影评论"""
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT c.id, c.user_id, u.username, c.movie_id,
                       c.content, c.created_at
                FROM comments c
                JOIN users u ON c.user_id = u.user_id
                WHERE c.movie_id = %s
                ORDER BY c.created_at DESC
                LIMIT 50
            """, (movie_id,))
            rows = cursor.fetchall()
        conn.close()

        comments = []
        for row in rows:
            comments.append({
                "id":         row["id"],
                "user_id":    row["user_id"],
                "username":   row["username"],
                "movie_id":   row["movie_id"],
                "content":    row["content"],
                "created_at": str(row["created_at"]),
            })
        return {"comments": comments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/movies/{movie_id}/comments")
async def add_comment(movie_id: int, req: CommentRequest):
    """发表评论"""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="评论内容不能为空")
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO comments (user_id, movie_id, content) VALUES (%s, %s, %s)",
                (req.user_id, movie_id, req.content.strip())
            )
            comment_id = cursor.lastrowid
            cursor.execute(
                "SELECT username FROM users WHERE user_id = %s", (req.user_id,)
            )
            user = cursor.fetchone()
        conn.commit()
        conn.close()
        return {
            "success": True,
            "comment": {
                "id":       comment_id,
                "user_id":  req.user_id,
                "username": user["username"] if user else f"user_{req.user_id}",
                "content":  req.content.strip(),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rate")
async def submit_rating(req: RatingRequest):
    if not 1.0 <= req.rating <= 5.0:
        raise HTTPException(status_code=400, detail="评分须在 1.0 ~ 5.0 之间")
    deleted = get_cache().invalidate_user(req.user_id)
    return {"success": True, "message": f"评分已提交，已清除 {deleted} 个旧缓存"}


@app.get("/cache/stats")
async def cache_stats():
    return get_cache().cache_stats()


@app.delete("/cache")
async def clear_all_cache():
    count = get_cache().invalidate_all()
    return {"message": f"已清除 {count} 个缓存 key"}




# ── A/B 测试接口 ──────────────────────────────

class ABClickRequest(BaseModel):
    user_id: int
    experiment_id: int
    variant: str
    movie_id: int


@app.get("/ab/variant/{user_id}")
async def get_ab_variant(user_id: int, experiment_id: int = 1):
    """
    获取用户的实验分组

    前端在请求推荐前先调用此接口，拿到分组后：
    1. 用对应的 mode 请求推荐
    2. 记录曝光事件
    """
    variant = _ab.get_variant(user_id, experiment_id)
    _ab.log_impression(user_id, experiment_id, variant)
    return {"variant": variant, "experiment_id": experiment_id}


@app.post("/ab/click")
async def log_ab_click(req: ABClickRequest):
    """记录用户点击事件"""
    _ab.log_click(req.user_id, req.experiment_id, req.variant, req.movie_id)
    return {"success": True}


@app.get("/ab/stats")
async def get_ab_stats(experiment_id: int = 1):
    """
    获取 A/B 实验统计结果

    返回各组 CTR 和统计显著性
    面试展示点：用数据驱动的方式验证推荐策略效果
    """
    return _ab.get_stats(experiment_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    