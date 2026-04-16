"""
models/collaborative.py
协同过滤：基于矩阵分解（SVD）的用户-物品推荐

核心思想：
  将评分矩阵 R(m×n) 分解为 U(m×k) · Σ · V^T(k×n)
  U：用户隐因子矩阵（每个用户 k 维偏好向量）
  V：物品隐因子矩阵（每个物品 k 维属性向量）
  通过隐向量内积预测用户对未看过电影的评分

面试要点：
1. SVD vs ALS vs SGD → 各自优缺点
2. 冷启动问题如何解决？→ 内容过滤兜底
3. 隐因子数 k 如何选择？→ 交叉验证
4. 如何加速推理？→ Faiss ANN 向量检索
"""
from __future__ import annotations
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class SVDRecommender:
    """
    基于截断 SVD 的矩阵分解推荐器

    优点：
    - 实现简单，易于解释
    - 天然支持 ANN 向量检索加速
    缺点：
    - 静态模型，新用户需重新训练（冷启动）
    - 内存占用随用户/物品数量增长
    """

    def __init__(self, n_factors: int = 50):
        """
        Args:
            n_factors: 隐因子维度 k，越大表达能力越强，但过拟合风险增加
                       经验值：50~200，用交叉验证确定
        """
        self.n_factors = n_factors
        self.user_factors: np.ndarray = None   # shape: (n_users, k)
        self.item_factors: np.ndarray = None   # shape: (n_items, k)
        self.user_bias: np.ndarray = None      # 用户偏差
        self.item_bias: np.ndarray = None      # 物品偏差
        self.global_mean: float = 0.0
        self._is_fitted = False

    def fit(self, interaction_matrix: csr_matrix) -> "SVDRecommender":
        """
        训练：对评分矩阵做截断 SVD 分解

        关键步骤：
        1. 计算全局均值、用户偏差、物品偏差（去除系统性偏差）
        2. 构建去偏后的评分矩阵
        3. 截断 SVD 分解
        4. 用奇异值加权用户/物品向量

        Args:
            interaction_matrix: CSR 稀疏评分矩阵 (n_users × n_items)
        """
        print(f"[SVD] 开始训练，隐因子维度 k={self.n_factors}...")

        # ── 1. 计算偏差项 ─────────────────────────────────────────
        # 面试：为什么要减去偏差？
        # → 有些用户习惯打高分，有些电影普遍被高分评价
        # → 去掉这部分系统性偏差，让模型专注学习"个性化偏好"
        matrix = interaction_matrix.astype(np.float64)
        nonzero_mask = matrix.copy()
        nonzero_mask.data[:] = 1.0  # 非零位置标记为1

        # 全局均值
        self.global_mean = matrix.data.mean() if len(matrix.data) > 0 else 3.5

        # 用户偏差 = 该用户平均评分 - 全局均值
        user_sum = np.array(matrix.sum(axis=1)).flatten()
        user_count = np.array(nonzero_mask.sum(axis=1)).flatten()
        user_count = np.where(user_count == 0, 1, user_count)  # 防止除零
        self.user_bias = user_sum / user_count - self.global_mean

        # 物品偏差 = 该物品平均评分 - 全局均值
        item_sum = np.array(matrix.sum(axis=0)).flatten()
        item_count = np.array(nonzero_mask.sum(axis=0)).flatten()
        item_count = np.where(item_count == 0, 1, item_count)
        self.item_bias = item_sum / item_count - self.global_mean

        # ── 2. 构建去偏矩阵（只在已评分位置减偏差）────────────────
        # 预测值 = global_mean + user_bias[u] + item_bias[i] + U[u]·V[i]
        # 所以这里先把前三项减掉，让 SVD 学习残差部分
        matrix_coo = matrix.tocoo()
        for idx in range(len(matrix_coo.data)):
            u = matrix_coo.row[idx]
            i = matrix_coo.col[idx]
            matrix_coo.data[idx] -= (
                self.global_mean + self.user_bias[u] + self.item_bias[i]
            )
        debiased_matrix = matrix_coo.tocsr()

        # ── 3. 截断 SVD 分解 ──────────────────────────────────────
        # R ≈ U · Σ · V^T
        # scipy svds 返回最大的 k 个奇异值
        k = min(self.n_factors, min(matrix.shape) - 1)
        U, sigma, Vt = svds(debiased_matrix, k=k)

        # ── 4. 用奇异值加权（将 Σ 融入 U 和 V）─────────────────────
        # 常见做法：U' = U·√Σ，V' = V·√Σ
        # 这样 U'·V'^T = U·Σ·V^T，方便直接用内积预测评分
        sigma_sqrt = np.sqrt(np.abs(sigma))
        self.user_factors = U * sigma_sqrt       # (n_users, k)
        self.item_factors = (Vt.T * sigma_sqrt)  # (n_items, k)

        self._is_fitted = True
        print(f"[SVD] 训练完成. 用户矩阵: {self.user_factors.shape}, "
              f"物品矩阵: {self.item_factors.shape}")
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        预测单个 (用户, 物品) 的评分

        预测公式：r̂(u,i) = μ + b_u + b_i + U_u · V_i
        """
        assert self._is_fitted, "请先调用 fit() 训练模型"
        score = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + self.user_factors[user_idx] @ self.item_factors[item_idx]
        )
        return float(np.clip(score, 1.0, 5.0))

    def recommend(
        self,
        user_idx: int,
        interaction_matrix: csr_matrix,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> list[tuple[int, float]]:
        """
        为用户生成 Top-K 推荐列表

        面试优化点：
        - 当 n_items 很大时，逐一计算 score 太慢
        - 优化：user_factors[u] · item_factors.T → 一次矩阵乘法得所有分数
        - 进一步优化：Faiss HNSW 近似最近邻，O(logN) 查询

        Args:
            user_idx: 用户索引
            interaction_matrix: 用于过滤已看过的电影
            top_k: 返回前 K 个推荐
            exclude_seen: 是否过滤已评分电影
        Returns:
            [(item_idx, predicted_score), ...] 按分数降序
        """
        assert self._is_fitted, "请先调用 fit() 训练模型"

        # 一次性计算该用户对所有电影的预测分数（矩阵乘法）
        user_vec = self.user_factors[user_idx]  # shape: (k,)
        all_scores = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.item_factors @ user_vec  # shape: (n_items,)
        )
        all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=5.0, neginf=0.0)
        all_scores = np.clip(all_scores, 0.0, 5.0)

        # 过滤已评分的电影
        if exclude_seen:
            seen_items = interaction_matrix[user_idx].nonzero()[1]
            all_scores[seen_items] = -np.inf

        # 取 Top-K
        actual_k = min(top_k, len(all_scores) - 1)
        top_indices = np.argpartition(all_scores, -actual_k)[-actual_k:]
        top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]

        return [(int(idx), float(all_scores[idx])) for idx in top_indices]

    def get_similar_items(self, item_idx: int, top_k: int = 10) -> list[tuple[int, float]]:
        """
        基于物品隐向量余弦相似度，找相似电影

        应用：详情页"相似电影推荐"
        """
        assert self._is_fitted, "请先调用 fit() 训练模型"

        item_vec = self.item_factors[item_idx]
        # 归一化后内积 = 余弦相似度
        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normalized = self.item_factors / norms
        item_vec_norm = item_vec / (np.linalg.norm(item_vec) + 1e-8)

        similarities = normalized @ item_vec_norm
        similarities[item_idx] = -1  # 排除自身

        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[SVD] 模型已保存至 {path}")

    @classmethod
    def load(cls, path: str) -> "SVDRecommender":
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[SVD] 模型已从 {path} 加载")
        return model


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.data_loader import generate_mock_data, DataProcessor

    ratings, movies = generate_mock_data(n_users=500, n_movies=30)
    processor = DataProcessor()
    train_df, test_df = processor.fit_transform(ratings)
    matrix = processor.build_interaction_matrix(train_df)

    model = SVDRecommender(n_factors=20)
    model.fit(matrix)

    # 为用户 0 推荐
    recs = model.recommend(user_idx=0, interaction_matrix=matrix, top_k=5)
    print(f"\n[用户0] Top-5 推荐:")
    for item_idx, score in recs:
        movie_id = processor.idx2item[item_idx]
        title = movies[movies.movie_id == movie_id].title.values[0]
        print(f"  {title}: 预测评分 {score:.2f}")
