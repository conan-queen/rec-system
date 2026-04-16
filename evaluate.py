"""
evaluate.py
推荐系统离线评估指标

面试必考：
1. RMSE vs MAE → RMSE 对大误差惩罚更重（平方放大）
2. Precision@K vs Recall@K → 精度 vs 覆盖
3. NDCG@K → 考虑排序位置的质量指标（位置越靠前权重越大）
4. Coverage & Diversity → 系统级别指标（不只看单用户准确率）
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. 评分预测类指标（Rating Prediction）
# ─────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    均方根误差（Root Mean Square Error）

    面试：RMSE 的缺点？
    → 对极端误差非常敏感（离群值会严重拉高 RMSE）
    → 不直接反映推荐排序质量
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差（更鲁棒于离群值）"""
    return float(np.mean(np.abs(y_true - y_pred)))


# ─────────────────────────────────────────────
# 2. Top-K 推荐类指标（Ranking Quality）
# ─────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Precision@K：前 K 个推荐中有多少是用户真正喜欢的

    公式：|推荐∩相关| / K
    适用：关注推荐精确度（宁缺毋滥）
    """
    if k == 0 or not recommended:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Recall@K：用户喜欢的电影中，有多少被推荐出来了

    公式：|推荐∩相关| / |相关|
    适用：关注覆盖率（不能漏推）
    """
    if not relevant or not recommended:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    NDCG@K（归一化折扣累积增益）

    核心思想：越靠前的位置越重要，相关结果排名越靠前得分越高

    公式：DCG@K / IDCG@K
    - DCG@K = Σ rel_i / log2(i+1)，i 从 1 开始
    - IDCG@K = 理想排序下的 DCG（所有相关结果排在前面）

    面试：为什么用 NDCG 而不是 Precision@K？
    → NDCG 考虑位置权重：第1位推对比第10位推对价值更高
    → NDCG 归一化到 [0,1]，方便跨用户/实验比较
    """
    if not relevant or not recommended:
        return 0.0

    # DCG：实际推荐序列
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)

    # IDCG：理想推荐序列（相关结果排在最前面）
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    HR@K（命中率）：推荐列表中是否至少命中1个相关物品

    最宽松的指标，适合评估系统底线能力
    """
    top_k = set(recommended[:k])
    return 1.0 if top_k & relevant else 0.0


# ─────────────────────────────────────────────
# 3. 全量评估（批量计算所有用户指标）
# ─────────────────────────────────────────────

def evaluate_recommender(
    model,
    test_df: pd.DataFrame,
    interaction_matrix: csr_matrix,
    processor,
    movies_df: pd.DataFrame,
    top_k: int = 10,
    rating_threshold: float = 3.5,
    n_users: int = 200   # 评估用户数（太大会慢）
) -> dict:
    """
    对模型进行全量离线评估

    Args:
        model: 实现了 recommend(user_idx, matrix, top_k) 方法的推荐器
        test_df: 测试集 DataFrame
        interaction_matrix: 训练集评分矩阵（用于过滤已看）
        processor: DataProcessor（含 idx2item 等映射）
        top_k: 推荐列表长度
        rating_threshold: 高于此分数算"相关"（正样本）

    Returns:
        {precision, recall, ndcg, hr, coverage} 平均值
    """
    # 构建测试集：每个用户的"相关电影"集合（高分电影）
    user_relevant = defaultdict(set)
    for _, row in test_df.iterrows():
        if row["rating"] >= rating_threshold:
            user_relevant[int(row["user_idx"])].add(int(row["item_idx"]))

    # 如果阈值太高导致没有正样本，降低阈值
    if len(user_relevant) == 0:
        for _, row in test_df.iterrows():
            user_relevant[int(row["user_idx"])].add(int(row["item_idx"]))

    test_users = list(user_relevant.keys())[:n_users]

    metrics = {
        "precision@k": [],
        "recall@k": [],
        "ndcg@k": [],
        "hit_rate@k": []
    }
    all_recommended_items = set()

    for user_idx in test_users:
        relevant = user_relevant[user_idx]
        if not relevant:
            continue

        try:
            # 获取推荐列表（item_idx 列表）
            recs = model.recommend(
                user_idx=user_idx,
                interaction_matrix=interaction_matrix,
                top_k=top_k
            )
            recommended = [r[0] if isinstance(r, tuple) else r["item_idx"] for r in recs]
        except Exception:
            continue

        all_recommended_items.update(recommended)

        metrics["precision@k"].append(precision_at_k(recommended, relevant, top_k))
        metrics["recall@k"].append(recall_at_k(recommended, relevant, top_k))
        metrics["ndcg@k"].append(ndcg_at_k(recommended, relevant, top_k))
        metrics["hit_rate@k"].append(hit_rate_at_k(recommended, relevant, top_k))

    # 系统级覆盖率：推荐覆盖了多少不同电影（多样性指标）
    catalog_size = processor.n_items
    coverage = len(all_recommended_items) / catalog_size if catalog_size > 0 else 0

    results = {
        f"precision@{top_k}": float(np.mean(metrics["precision@k"])),
        f"recall@{top_k}": float(np.mean(metrics["recall@k"])),
        f"ndcg@{top_k}": float(np.mean(metrics["ndcg@k"])),
        f"hit_rate@{top_k}": float(np.mean(metrics["hit_rate@k"])),
        "coverage": coverage,
        "n_users_evaluated": len(metrics["precision@k"])
    }

    return results


def evaluate_rating_prediction(
    model,
    test_df: pd.DataFrame,
    interaction_matrix: csr_matrix
) -> dict:
    """
    评估评分预测准确率（RMSE/MAE）

    面试：什么时候用 RMSE，什么时候用 Ranking 指标？
    → 如果产品形态是"展示预测评分"→ 用 RMSE
    → 如果产品是"推荐列表"→ 用 NDCG/Precision（更贴近实际）
    """
    y_true, y_pred = [], []

    for _, row in test_df.iterrows():
        u, i = int(row["user_idx"]), int(row["item_idx"])
        if u >= model.user_factors.shape[0] or i >= model.item_factors.shape[0]:
            continue
        pred = model.predict(u, i)
        y_true.append(row["rating"])
        y_pred.append(pred)

    if not y_true:
        return {"rmse": None, "mae": None}

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    return {
        "rmse": rmse(y_true_arr, y_pred_arr),
        "mae": mae(y_true_arr, y_pred_arr),
        "n_predictions": len(y_true)
    }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data.data_loader import generate_mock_data, DataProcessor
    from models.collaborative import SVDRecommender

    ratings, movies = generate_mock_data(n_users=500, n_movies=30)
    processor = DataProcessor()
    train_df, test_df = processor.fit_transform(ratings)
    matrix = processor.build_interaction_matrix(train_df)

    model = SVDRecommender(n_factors=20).fit(matrix)

    # 评分预测指标
    rating_metrics = evaluate_rating_prediction(model, test_df, matrix)
    print("\n[评分预测指标]")
    print(f"  RMSE: {rating_metrics['rmse']:.4f}")
    print(f"  MAE:  {rating_metrics['mae']:.4f}")

    # Top-K 推荐指标
    ranking_metrics = evaluate_recommender(
        model, test_df, matrix, processor, movies, top_k=10
    )
    print("\n[Top-K 推荐指标]")
    for k, v in ranking_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
