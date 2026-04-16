"""
feature_store/feature_store.py
Feature Store：集中管理用户和电影特征

核心价值：
1. 解决 Training-Serving Skew：训练和推理用同一套特征
2. 特征复用：多个模型共享同一套特征，不重复计算
3. 低延迟：特征存在 Redis，推理时毫秒级读取
4. 版本管理：特征更新时保留历史版本，方便回滚

面试要点：
Q: Feature Store 解决了什么问题？
A: 训练/服务一致性（最重要）、特征重复开发、版本管理

Q: 为什么用 Redis 存特征？
A: 低延迟（<1ms）、支持高并发读取、TTL 自动过期

Q: 特征更新频率是多少？
A: 用户特征每天更新（批处理）；实时特征（最近点击）每次行为后更新

Q: 如果 Redis 挂了怎么办？
A: 降级到实时计算，性能下降但服务不中断
"""
from __future__ import annotations

import os
import json
import math


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


def get_redis():
    import redis
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=True,
        socket_timeout=1,
    )


class FeatureStore:
    """
    Feature Store 核心类

    特征分类：
    ┌─────────────────────────────────────────┐
    │  用户特征（User Features）               │
    │  - avg_rating: 用户平均评分              │
    │  - rating_count: 评分总数（活跃度）      │
    │  - favorite_genres: 最喜欢的类型 Top-3   │
    │  - rating_std: 评分标准差（口味一致性）  │
    ├─────────────────────────────────────────┤
    │  电影特征（Item Features）               │
    │  - avg_rating: 电影平均评分              │
    │  - rating_count: 被评分次数（热度）      │
    │  - genres_list: 类型列表                 │
    │  - popularity_score: 热度分              │
    └─────────────────────────────────────────┘
    """

    USER_FEATURE_TTL = 86400      # 用户特征 24 小时过期
    ITEM_FEATURE_TTL = 3600 * 6   # 电影特征 6 小时过期
    USER_KEY_PREFIX = "fs:user:"
    ITEM_KEY_PREFIX = "fs:item:"

    def __init__(self):
        self._redis = None
        self._redis_available = False
        self._connect_redis()

    def _connect_redis(self):
        try:
            r = get_redis()
            r.ping()
            self._redis = r
            self._redis_available = True
            print("[Feature Store] Redis 连接成功")
        except Exception as e:
            self._redis_available = False
            print(f"[Feature Store] Redis 不可用，将实时计算特征: {e}")

    # ── 用户特征 ────────────────────────────────────────────

    def get_user_features(self, user_id: int) -> dict:
        """
        获取用户特征

        优先从 Redis 读取（缓存命中 < 1ms）
        缓存未命中则实时从 MySQL 计算并写入缓存
        """
        # 1. 尝试从 Redis 读取
        if self._redis_available:
            try:
                key = f"{self.USER_KEY_PREFIX}{user_id}"
                cached = self._redis.get(key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        # 2. 缓存未命中，实时计算
        features = self._compute_user_features(user_id)

        # 3. 写入 Redis 缓存
        if self._redis_available and features:
            try:
                key = f"{self.USER_KEY_PREFIX}{user_id}"
                self._redis.setex(key, self.USER_FEATURE_TTL, json.dumps(features))
            except Exception:
                pass

        return features

    def _compute_user_features(self, user_id: int) -> dict:
        """从 MySQL 实时计算用户特征"""
        try:
            conn = get_db()
            with conn.cursor() as cursor:
                # 基础统计特征
                cursor.execute("""
                    SELECT
                        COUNT(*)           AS rating_count,
                        AVG(rating)        AS avg_rating,
                        STDDEV(rating)     AS rating_std,
                        MIN(rating)        AS min_rating,
                        MAX(rating)        AS max_rating
                    FROM ratings
                    WHERE user_id = %s
                """, (user_id,))
                stats = cursor.fetchone()

                # 偏好类型（评分最高的类型）
                cursor.execute("""
                    SELECT m.genres, AVG(r.rating) AS avg_r, COUNT(*) AS cnt
                    FROM ratings r
                    JOIN movies m ON r.movie_id = m.movie_id
                    WHERE r.user_id = %s AND r.rating >= 3.5
                    GROUP BY m.genres
                    ORDER BY avg_r DESC
                    LIMIT 10
                """, (user_id,))
                genre_rows = cursor.fetchall()

            conn.close()

            # 统计偏好类型
            genre_counts = {}
            for row in genre_rows:
                for g in (row["genres"] or "").split("|"):
                    g = g.strip()
                    if g:
                        genre_counts[g] = genre_counts.get(g, 0) + row["cnt"]

            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            return {
                "user_id":       user_id,
                "rating_count":  int(stats["rating_count"] or 0),
                "avg_rating":    round(float(stats["avg_rating"] or 0), 3),
                "rating_std":    round(float(stats["rating_std"] or 0), 3),
                "min_rating":    float(stats["min_rating"] or 0),
                "max_rating":    float(stats["max_rating"] or 0),
                "favorite_genres": [g for g, _ in top_genres],
                "user_tier":     self._get_user_tier(int(stats["rating_count"] or 0)),
            }
        except Exception as e:
            print(f"[Feature Store] 计算用户特征失败: {e}")
            return {}

    @staticmethod
    def _get_user_tier(rating_count: int) -> str:
        """
        用户分层：根据活跃度分层
        用于 A/B 实验的分层分析，避免辛普森悖论
        """
        if rating_count >= 50:
            return "power"     # 重度用户
        elif rating_count >= 10:
            return "active"    # 活跃用户
        elif rating_count >= 1:
            return "casual"    # 轻度用户
        else:
            return "cold"      # 冷启动用户

    # ── 电影特征 ────────────────────────────────────────────

    def get_item_features(self, movie_id: int) -> dict:
        """获取电影特征（优先 Redis，降级实时计算）"""
        if self._redis_available:
            try:
                key = f"{self.ITEM_KEY_PREFIX}{movie_id}"
                cached = self._redis.get(key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        features = self._compute_item_features(movie_id)

        if self._redis_available and features:
            try:
                key = f"{self.ITEM_KEY_PREFIX}{movie_id}"
                self._redis.setex(key, self.ITEM_FEATURE_TTL, json.dumps(features))
            except Exception:
                pass

        return features

    def _compute_item_features(self, movie_id: int) -> dict:
        """从 MySQL 实时计算电影特征"""
        try:
            conn = get_db()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        m.title, m.genres, m.year,
                        COUNT(r.id)        AS rating_count,
                        AVG(r.rating)      AS avg_rating,
                        STDDEV(r.rating)   AS rating_std
                    FROM movies m
                    LEFT JOIN ratings r ON m.movie_id = r.movie_id
                    WHERE m.movie_id = %s
                    GROUP BY m.movie_id
                """, (movie_id,))
                row = cursor.fetchone()
            conn.close()

            if not row:
                return {}

            rating_count = int(row["rating_count"] or 0)
            avg_rating   = float(row["avg_rating"] or 0)

            # 热度分：综合评分数量和评分高低
            # 公式：贝叶斯平均（避免评分数少的电影虚高）
            # score = (v * R + m * C) / (v + m)
            # v=评分数, R=平均分, m=最小评分数阈值, C=全局均值
            m_threshold = 10
            global_mean = 3.5
            popularity = (rating_count * avg_rating + m_threshold * global_mean) / (rating_count + m_threshold)

            return {
                "movie_id":        movie_id,
                "title":           row["title"],
                "genres":          (row["genres"] or "").split("|"),
                "year":            row["year"],
                "rating_count":    rating_count,
                "avg_rating":      round(avg_rating, 3),
                "rating_std":      round(float(row["rating_std"] or 0), 3),
                "popularity_score": round(popularity, 3),
            }
        except Exception as e:
            print(f"[Feature Store] 计算电影特征失败: {e}")
            return {}

    # ── 批量更新 ────────────────────────────────────────────

    def warm_up_cache(self, user_ids: list, movie_ids: list) -> None:
        """
        预热缓存：启动时批量计算并缓存高活跃用户的特征

        场景：
        - 服务启动时预热 Top-100 活跃用户的特征
        - 避免冷启动时大量 Cache Miss 导致延迟高峰

        面试：这叫 Cache Warm-up / Pre-computation
        """
        print(f"[Feature Store] 预热缓存: {len(user_ids)} 用户, {len(movie_ids)} 电影")
        for uid in user_ids:
            self.get_user_features(uid)
        for mid in movie_ids:
            self.get_item_features(mid)
        print("[Feature Store] 缓存预热完成")

    def invalidate_user(self, user_id: int) -> None:
        """用户产生新行为时，使其特征缓存失效"""
        if self._redis_available:
            try:
                self._redis.delete(f"{self.USER_KEY_PREFIX}{user_id}")
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Feature Store 统计信息"""
        if not self._redis_available:
            return {"available": False}
        try:
            user_keys = len(self._redis.keys(f"{self.USER_KEY_PREFIX}*"))
            item_keys = len(self._redis.keys(f"{self.ITEM_KEY_PREFIX}*"))
            return {
                "available":         True,
                "cached_users":      user_keys,
                "cached_items":      item_keys,
                "user_ttl_hours":    self.USER_FEATURE_TTL // 3600,
                "item_ttl_hours":    self.ITEM_FEATURE_TTL // 3600,
            }
        except Exception:
            return {"available": False}
