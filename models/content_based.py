"""
models/content_based.py
内容过滤：基于电影类型/描述的 TF-IDF 相似度推荐

核心思想：
  将电影的文本特征（类型、描述）转为 TF-IDF 向量
  计算电影间的余弦相似度
  基于用户历史喜好，找相似电影推荐

面试要点：
1. 优点：解决冷启动（新电影可以立即推荐）
2. 缺点：信息茧房（只推和历史相似的，缺乏探索）
3. 为什么用 TF-IDF 而不是 one-hot？→ 降低高频类型的权重
4. 优化：用预训练词向量（Word2Vec/BERT）替代 TF-IDF
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle


def _genre_tokenizer(x):
    """模块级 tokenizer，支持 pickle 序列化"""
    return x.split("|")


class ContentBasedRecommender:
    """
    基于内容的推荐器

    特征来源：
    - 电影类型（genres）：Action|Comedy|Drama
    - 发行年份（year）
    - 可扩展：导演、演员、剧情简介（需额外数据）
    """

    def __init__(self, n_recommendations: int = 10):
        self.n_recommendations = n_recommendations
        self.tfidf = TfidfVectorizer(
            tokenizer=_genre_tokenizer,  # genres 用 | 分隔
            token_pattern=None,
            lowercase=False
        )
        self.item_profiles: np.ndarray = None   # TF-IDF 特征矩阵
        self.similarity_matrix: np.ndarray = None  # 电影间相似度矩阵
        self.movie_ids: list = []
        self._is_fitted = False

    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        构建电影内容特征向量并计算相似度矩阵

        Args:
            movies_df: 包含 [movie_id, title, genres, year] 的 DataFrame
        """
        print(f"[内容过滤] 开始构建电影特征，共 {len(movies_df)} 部电影...")

        self.movie_ids = movies_df["movie_id"].tolist()
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

        # ── 1. 特征工程 ────────────────────────────────────────────
        # 组合类型 + 年代段（可扩展更多特征）
        features = movies_df.apply(self._build_feature_text, axis=1)

        # ── 2. TF-IDF 向量化 ───────────────────────────────────────
        # genres 已是 | 分隔，直接用自定义 tokenizer
        self.item_profiles = self.tfidf.fit_transform(features)
        print(f"[内容过滤] TF-IDF 矩阵 shape: {self.item_profiles.shape}")

        # ── 3. 计算物品间相似度矩阵 ────────────────────────────────
        # 注意：n_items 很大时不能存全矩阵（内存爆炸）
        # 生产中改用 ANN（Faiss/ScaNN）查询最近邻
        self.similarity_matrix = cosine_similarity(self.item_profiles)
        np.fill_diagonal(self.similarity_matrix, 0)  # 自身相似度不用

        self._is_fitted = True
        print("[内容过滤] 训练完成")
        return self

    def _build_feature_text(self, row: pd.Series) -> str:
        """
        将电影一行记录转为特征文本字符串

        面试延伸：实际项目中可加入：
        - 导演名（director_Action → 自定义 token）
        - 演员列表
        - 剧情关键词（BERT embedding 更好）
        - 用户标签（tag-genome）
        """
        parts = [row["genres"]]
        # 年代段特征（decade token）
        if "year" in row and pd.notna(row.get("year")):
            decade = f"decade_{int(row['year']) // 10 * 10}s"
            parts.append(decade)
        return "|".join(parts)

    def get_similar_movies(
        self,
        movie_id: int,
        top_k: int = 10
    ) -> list[tuple[int, float]]:
        """
        找与指定电影最相似的 Top-K 电影

        用途：详情页"你可能还喜欢"、冷启动推荐

        Returns:
            [(movie_id, similarity_score), ...] 降序排列
        """
        assert self._is_fitted, "请先调用 fit()"

        idx = self.movie_id_to_idx.get(movie_id)
        if idx is None:
            return []

        scores = self.similarity_matrix[idx]
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.movie_ids[i], float(scores[i])) for i in top_indices]

    def recommend_for_user(
        self,
        user_id: int,
        interaction_matrix: csr_matrix,
        user_idx: int,
        item2idx: dict,
        idx2item: dict,
        top_k: int = 10
    ) -> list[tuple[int, float]]:
        """
        基于用户历史评分行为，推荐内容相似的电影

        策略：
        1. 找到用户评分最高的 N 部电影（用户口味锚点）
        2. 对每部高分电影找相似电影
        3. 汇总相似分数（加权平均）
        4. 过滤已看，返回 Top-K

        面试要点：
        - 为什么不用所有历史记录而用"高分电影"？
          → 用户低分电影说明"不喜欢"，用它找相似反而有害
        - 相似分数如何聚合？→ 加权平均（权重=原始评分）
        """
        assert self._is_fitted, "请先调用 fit()"

        # 获取用户已评分电影
        user_row = interaction_matrix.getrow(user_idx)
        rated_items = user_row.nonzero()[1]
        if len(rated_items) == 0:
            # 冷启动：返回热门电影（按平均相似度排序）
            avg_sim = self.similarity_matrix.mean(axis=0)
            top_k_idx = np.argsort(avg_sim)[::-1][:top_k]
            return [(self.movie_ids[i], float(avg_sim[i])) for i in top_k_idx]

        # 提取评分（转为 1D numpy 数组，避免稀疏矩阵索引返回 matrix 对象）
        user_dense = user_row.toarray().flatten()   # shape: (n_items,)
        ratings_arr = user_dense[rated_items]       # shape: (n_rated,)

        # 只用评分 >= 3.5 的电影作为口味锚点
        liked_mask = ratings_arr >= 3.5
        if liked_mask.sum() == 0:
            liked_mask = ratings_arr >= ratings_arr.mean()

        liked_items = rated_items[liked_mask]
        liked_ratings = ratings_arr[liked_mask]

        # 聚合相似分数
        score_accumulator = np.zeros(len(self.movie_ids))
        weight_sum = 0.0

        for item_idx, rating in zip(liked_items, liked_ratings):
            movie_id = idx2item.get(item_idx)
            if movie_id is None:
                continue
            movie_profile_idx = self.movie_id_to_idx.get(movie_id)
            if movie_profile_idx is None:
                continue
            # 加权：评分越高的电影贡献越大
            weight = rating - 2.5  # 偏好权重（高于均值才为正）
            if weight > 0:
                score_accumulator += self.similarity_matrix[movie_profile_idx] * weight
                weight_sum += weight

        if weight_sum > 0:
            score_accumulator /= weight_sum

        # 过滤已看
        seen_movie_ids = {idx2item.get(i) for i in rated_items}
        for mid in seen_movie_ids:
            if mid is not None and mid in self.movie_id_to_idx:
                score_accumulator[self.movie_id_to_idx[mid]] = -1

        top_indices = np.argpartition(score_accumulator, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(score_accumulator[top_indices])[::-1]]

        return [
            (self.movie_ids[i], float(score_accumulator[i]))
            for i in top_indices
            if score_accumulator[i] > 0
        ][:top_k]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[内容过滤] 模型已保存至 {path}")

    @classmethod
    def load(cls, path: str) -> "ContentBasedRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.data_loader import generate_mock_data, DataProcessor

    ratings, movies = generate_mock_data()
    processor = DataProcessor()
    train_df, test_df = processor.fit_transform(ratings)
    matrix = processor.build_interaction_matrix(train_df)

    model = ContentBasedRecommender()
    model.fit(movies)

    # 找与电影0相似的电影
    similar = model.get_similar_movies(movie_id=0, top_k=5)
    print("\n与电影0最相似的5部电影：")
    for mid, score in similar:
        title = movies[movies.movie_id == mid].title.values[0]
        genres = movies[movies.movie_id == mid].genres.values[0]
        print(f"  {title} ({genres}): {score:.3f}")
