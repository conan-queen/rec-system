"""
db/mysql_loader.py
从 MySQL 读取真实数据，替换原来的模拟数据

使用方式：
    原来：ratings_df, movies_df = generate_mock_data()
    现在：ratings_df, movies_df = MySQLLoader(config).load()
"""
from __future__ import annotations

import pandas as pd


# ── MySQL 连接配置 ─────────────────────────────
# 生产环境建议从环境变量读取，不要硬编码密码
import os

DEFAULT_CONFIG = {
    "host":     os.getenv("MYSQL_HOST",     "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER",     "root"),
    "password": os.getenv("MYSQL_PASSWORD", "yourpassword"),
    "database": os.getenv("MYSQL_DATABASE", "rec_system"),
}


class MySQLLoader:
    """
    从 MySQL 加载训练数据

    面试要点：
    - 为什么用环境变量存密码？→ 避免密码提交到 Git，安全规范
    - 大数据量时如何优化？→ 分批读取（chunksize），避免内存溢出
    - 如何保证数据质量？→ 过滤异常评分，去除僵尸用户
    """

    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG

    def _get_connection(self):
        """建立 MySQL 连接"""
        try:
            import pymysql
        except ImportError:
            raise ImportError("请先安装：pip install pymysql")

        return pymysql.connect(
            host=self.config["host"],
            port=self.config["port"],
            user=self.config["user"],
            password=self.config["password"],
            database=self.config["database"],
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT user_id, movie_id, rating,
                           UNIX_TIMESTAMP(created_at) AS timestamp
                    FROM ratings
                    WHERE rating BETWEEN 1.0 AND 5.0
                    ORDER BY created_at ASC
                """)
                ratings_rows = cursor.fetchall()

                cursor.execute("""
                    SELECT movie_id, title, genres, year
                    FROM movies
                """)
                movies_rows = cursor.fetchall()

            ratings_df = pd.DataFrame(ratings_rows)
            movies_df = pd.DataFrame(movies_rows)

            ratings_df["user_id"] = ratings_df["user_id"].astype(int)
            ratings_df["movie_id"] = ratings_df["movie_id"].astype(int)
            ratings_df["rating"] = ratings_df["rating"].astype(float)
            ratings_df["timestamp"] = ratings_df["timestamp"].astype(int)
            movies_df["movie_id"] = movies_df["movie_id"].astype(int)

            print(f"[MySQL] 加载评分: {len(ratings_df)} 条, "
                  f"电影: {len(movies_df)} 部, "
                  f"用户: {ratings_df['user_id'].nunique()} 个")

            return ratings_df, movies_df

        finally:
            conn.close()
            
    def save_ratings(self, ratings: list[dict]) -> None:
        """
        写入新的用户评分（实时收集用户行为时调用）

        Args:
            ratings: [{"user_id": 1, "movie_id": 3, "rating": 4.5}, ...]
        """
        if not ratings:
            return

        conn = self._get_connection()
        try:
            import pymysql
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO ratings (user_id, movie_id, rating)
                    VALUES (%(user_id)s, %(movie_id)s, %(rating)s)
                    ON DUPLICATE KEY UPDATE
                        rating = VALUES(rating),
                        created_at = CURRENT_TIMESTAMP
                """
                cursor.executemany(sql, ratings)
            conn.commit()
            print(f"[MySQL] 写入 {len(ratings)} 条评分")
        finally:
            conn.close()

    def log_recommendations(self, logs: list[dict]) -> None:
        """
        记录推荐日志（用于后续效果分析）

        Args:
            logs: [{"user_id":1, "movie_id":5, "score":0.8,
                    "source":"hybrid", "position":1}, ...]
        """
        if not logs:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO recommendation_logs
                        (user_id, movie_id, score, source, position)
                    VALUES
                        (%(user_id)s, %(movie_id)s, %(score)s,
                         %(source)s, %(position)s)
                """
                cursor.executemany(sql, logs)
            conn.commit()
        finally:
            conn.close()


def test_connection(config: dict = None) -> bool:
    """测试数据库连接是否正常"""
    try:
        loader = MySQLLoader(config)
        conn = loader._get_connection()
        conn.close()
        print("[MySQL] 连接成功")
        return True
    except Exception as e:
        print(f"[MySQL] 连接失败: {e}")
        return False
