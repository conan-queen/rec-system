"""
train.py
训练入口：一键训练所有模型并评估

运行方式：
    cd rec_system
    python train.py
"""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import generate_mock_data, DataProcessor
from models.collaborative import SVDRecommender
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender
from evaluate import evaluate_recommender, evaluate_rating_prediction


def train_and_evaluate():
    print("=" * 55)
    print("  电影推荐系统 - 训练与评估")
    print("=" * 55)

    # ── 1. 数据准备 ────────────────────────────────────────────────
    print("\n【Step 1】加载数据...")
    USE_MYSQL = os.getenv("USE_MYSQL", "false").lower() == "true"

    if USE_MYSQL:
        from db.mysql_loader import MySQLLoader
        ratings_df, movies_df = MySQLLoader().load()
        print("[数据] 使用 MySQL 真实数据")
    else:
        ratings_df, movies_df = generate_mock_data(
            n_users=1000, n_movies=30, rating_density=0.20
        )
        print("[数据] 使用模拟数据")
    processor = DataProcessor()
    train_df, test_df = processor.fit_transform(ratings_df, test_ratio=0.2)
    interaction_matrix = processor.build_interaction_matrix(train_df)

    # ── 2. 训练协同过滤模型 ────────────────────────────────────────
    print("\n【Step 2】训练协同过滤模型（SVD Matrix Factorization）...")
    cf_model = SVDRecommender(n_factors=50)
    cf_model.fit(interaction_matrix)

    # ── 3. 训练内容过滤模型 ────────────────────────────────────────
    print("\n【Step 3】训练内容过滤模型（TF-IDF + Cosine Similarity）...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(movies_df)

    # ── 4. 构建混合模型 ────────────────────────────────────────────
    print("\n【Step 4】构建混合推荐模型...")
    hybrid_model = HybridRecommender(cf_model=cf_model, cb_model=cb_model)

    # ── 5. 评分预测评估（CF 专属）─────────────────────────────────
    print("\n【Step 5】离线评估...")
    print("\n  [协同过滤 - 评分预测指标]")
    rating_metrics = evaluate_rating_prediction(cf_model, test_df, interaction_matrix)
    print(f"    RMSE : {rating_metrics['rmse']:.4f}")
    print(f"    MAE  : {rating_metrics['mae']:.4f}")
    print(f"    样本数: {rating_metrics['n_predictions']}")

    # ── 6. Top-K 推荐排序评估 ──────────────────────────────────────
    print("\n  [协同过滤 - Top-10 排序指标]")
    cf_ranking = evaluate_recommender(
        cf_model, test_df, interaction_matrix, processor, movies_df,
        top_k=10, n_users=200
    )
    for k, v in cf_ranking.items():
        print(f"    {k:<20}: {v:.4f}" if isinstance(v, float) else f"    {k:<20}: {v}")

    # ── 7. 展示推荐样例 ────────────────────────────────────────────
    print("\n【Step 6】推荐样例展示...")
    for user_idx in [0, 1, 2]:
        n_hist = interaction_matrix[user_idx].nnz
        print(f"\n  [用户 {user_idx}] 历史行为: {n_hist} 条")

        hybrid_recs = hybrid_model.recommend(
            user_idx=user_idx,
            interaction_matrix=interaction_matrix,
            item2idx=processor.item2idx,
            idx2item=processor.idx2item,
            top_k=5
        )

        for rank, r in enumerate(hybrid_recs, 1):
            movie_id = processor.idx2item.get(r["item_idx"])
            if movie_id is None:
                continue
            row = movies_df[movies_df.movie_id == movie_id]
            if row.empty:
                continue
            title = row.title.values[0]
            genres = row.genres.values[0]
            print(f"    {rank}. {title:<30} {genres:<25} "
                  f"分={r['score']:.3f} [{r['source']}]")

    # ── 8. 保存模型 ────────────────────────────────────────────────
    os.makedirs("saved_models", exist_ok=True)
    cf_model.save("saved_models/cf_model.pkl")
    cb_model.save("saved_models/cb_model.pkl")
    print("\n[完成] 模型已保存至 saved_models/")

    # 保存 processor（API 层需要用到 idx 映射）
    import pickle
    with open("saved_models/processor.pkl", "wb") as f:
        pickle.dump(processor, f)
    with open("saved_models/movies.pkl", "wb") as f:
        pickle.dump(movies_df, f)
    with open("saved_models/matrix.pkl", "wb") as f:
        pickle.dump(interaction_matrix, f)

    print("\n" + "=" * 55)
    print("  训练完成！可以运行 API 服务：")
    print("  uvicorn api.server:app --reload")
    print("=" * 55)


if __name__ == "__main__":
    train_and_evaluate()
