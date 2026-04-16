"""
data/data_loader.py
数据加载与预处理模块

面试要点：
- 为什么要 split 时按时间排序？→ 避免数据泄露（leakage）
- 为什么做评分归一化？→ 消除用户评分偏差（有人普遍打高分）
- 稀疏矩阵用什么格式？→ scipy.sparse.csr_matrix，节省内存
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 模拟 MovieLens 数据集（实际项目替换为真实数据）
# MovieLens 官方地址：https://grouplens.org/datasets/movielens/
# ─────────────────────────────────────────────

GENRES = [
    "Action", "Comedy", "Drama", "Romance", "Thriller",
    "Sci-Fi", "Horror", "Animation", "Documentary", "Adventure"
]

MOVIE_TITLES = [
    "Inception", "The Godfather", "Interstellar", "Toy Story",
    "The Dark Knight", "Forrest Gump", "Pulp Fiction", "The Matrix",
    "Schindler's List", "Goodfellas", "Fight Club", "The Silence of the Lambs",
    "Saving Private Ryan", "Gladiator", "The Shawshank Redemption",
    "Avengers: Endgame", "Parasite", "Your Name", "Spirited Away",
    "The Lion King", "Titanic", "Avatar", "Jurassic Park", "Home Alone",
    "Die Hard", "Star Wars", "The Lord of the Rings", "Harry Potter",
    "Iron Man", "Spider-Man"
]


def generate_mock_data(
    n_users: int = 1000,
    n_movies: int = 30,
    rating_density: float = 0.15,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成模拟的用户评分数据和电影元数据。

    Returns:
        ratings_df: [user_id, movie_id, rating, timestamp]
        movies_df:  [movie_id, title, genres]
    """
    np.random.seed(random_seed)

    # ── 电影元数据 ──────────────────────────────
    movies_data = []
    for i in range(n_movies):
        # 每部电影随机分配 1~3 个类型
        n_genres = np.random.randint(1, 4)
        selected_genres = "|".join(np.random.choice(GENRES, n_genres, replace=False))
        movies_data.append({
            "movie_id": i,
            "title": MOVIE_TITLES[i % len(MOVIE_TITLES)],
            "genres": selected_genres,
            "year": np.random.randint(1990, 2024)
        })
    movies_df = pd.DataFrame(movies_data)

    # ── 用户评分 ────────────────────────────────
    # 模拟真实场景：不同用户的评分偏好不同（user bias）
    user_biases = np.random.normal(0, 0.5, n_users)   # 用户偏差
    movie_biases = np.random.normal(0, 0.5, n_movies)  # 电影偏差

    ratings_data = []
    for user_id in range(n_users):
        # 每个用户随机评分 density * n_movies 部电影
        n_rated = max(1, int(n_movies * rating_density))
        rated_movies = np.random.choice(n_movies, n_rated, replace=False)
        for movie_id in rated_movies:
            # 评分 = 3.5（均值）+ 用户偏差 + 电影偏差 + 随机噪声
            base_rating = 3.5 + user_biases[user_id] + movie_biases[movie_id]
            noise = np.random.normal(0, 0.3)
            rating = np.clip(round(base_rating + noise) * 0.5, 1.0, 5.0) * 2 / 2
            timestamp = np.random.randint(1_000_000_000, 1_700_000_000)
            ratings_data.append({
                "user_id": user_id,
                "movie_id": int(movie_id),
                "rating": rating,
                "timestamp": timestamp
            })

    ratings_df = pd.DataFrame(ratings_data)
    return ratings_df, movies_df


class DataProcessor:
    """
    数据预处理器：负责训练/测试分割、用户/物品编码、构建稀疏矩阵。

    面试考点：
    - 为什么按时间排序分割？→ 模拟真实场景，避免未来数据泄露给过去
    - user/item 的 id 为什么需要重新编码？→ 原始 id 可能不连续，影响矩阵构建
    """

    def __init__(self):
        self.user2idx: dict = {}
        self.idx2user: dict = {}
        self.item2idx: dict = {}
        self.idx2item: dict = {}
        self.n_users: int = 0
        self.n_items: int = 0

    def fit_transform(
        self,
        ratings_df: pd.DataFrame,
        test_ratio: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        编码 + 时间序列划分训练/测试集

        Args:
            ratings_df: 原始评分表
            test_ratio: 测试集比例
        Returns:
            train_df, test_df（含编码后的 user_idx / item_idx）
        """
        # 1. 按时间排序（关键！避免数据泄露）
        df = ratings_df.sort_values("timestamp").reset_index(drop=True)

        # 2. 用户 & 物品 ID 编码（连续整数索引）
        users = df["user_id"].unique()
        items = df["movie_id"].unique()
        self.user2idx = {int(u): i for i, u in enumerate(users)}
        self.idx2user = {i: int(u) for u, i in self.user2idx.items()}
        self.item2idx = {int(it): i for i, it in enumerate(items)}
        self.idx2item = {i: int(it) for it, i in self.item2idx.items()}
        self.n_users = len(users)
        self.n_items = len(items)

        df["user_idx"] = df["user_id"].map(self.user2idx)
        df["item_idx"] = df["movie_id"].map(self.item2idx)

        # 3. 按时间顺序划分（每个用户最后 test_ratio 比例的评分放入测试集）
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"[数据] 用户数: {self.n_users}, 电影数: {self.n_items}")
        print(f"[数据] 训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")
        print(f"[数据] 稀疏度: {len(df) / (self.n_users * self.n_items):.2%}")

        return train_df, test_df

    def build_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """
        构建用户-物品稀疏评分矩阵（CSR 格式节省内存）

        面试：为什么用 CSR？→ 行切片高效，适合按用户查询
        """
        matrix = csr_matrix(
            (df["rating"].values, (df["user_idx"].values, df["item_idx"].values)),
            shape=(self.n_users, self.n_items)
        )
        return matrix


if __name__ == "__main__":
    # 快速验证数据加载
    ratings, movies = generate_mock_data()
    print(f"评分数据 shape: {ratings.shape}")
    print(ratings.head())
    print(f"\n电影数据 shape: {movies.shape}")
    print(movies.head())

    processor = DataProcessor()
    train, test = processor.fit_transform(ratings)
    matrix = processor.build_interaction_matrix(train)
    print(f"\n评分矩阵 shape: {matrix.shape}, 非零元素: {matrix.nnz}")
