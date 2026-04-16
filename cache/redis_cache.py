"""
cache/redis_cache.py
推荐结果 Redis 缓存

作用：
- 避免每次请求都重新计算推荐（SVD矩阵乘法虽然快，但高并发时仍是瓶颈）
- 活跃用户的推荐结果缓存 1 小时，冷门用户缓存 10 分钟
- 模型重训后自动清除所有缓存

面试要点：
1. 缓存 key 怎么设计？→ rec:{user_id}:{top_k}:{mode}，包含所有影响结果的参数
2. 缓存多久过期？→ 根据用户活跃度区分，活跃用户更新快所以 TTL 更短
3. 模型更新后怎么刷新缓存？→ 按前缀批量删除
4. 缓存穿透怎么处理？→ 不存在的 user_id 直接在 API 层拦截，不查缓存
"""
from __future__ import annotations

import numpy as np
import json
import os

REDIS_CONFIG = {
    "host":     os.getenv("REDIS_HOST",     "localhost"),
    "port":     int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("REDIS_PASSWORD", None),
    "db":       int(os.getenv("REDIS_DB",   "0")),
}

# 缓存过期时间（秒）
TTL_ACTIVE_USER  = 3600    # 活跃用户：1 小时
TTL_NORMAL_USER  = 600     # 普通用户：10 分钟
TTL_SIMILAR      = 86400   # 相似电影：24 小时（不依赖用户，变化更慢）

KEY_PREFIX = "rec"         # 所有推荐缓存的 key 前缀


class RecommendationCache:
    """
    推荐结果缓存器

    连接失败时自动降级（不影响主流程，只是不走缓存）
    """

    def __init__(self, config: dict = None):
        self.config = config or REDIS_CONFIG
        self._client = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        """建立 Redis 连接，失败时标记为不可用"""
        try:
            import redis
            self._client = redis.Redis(
                host=self.config["host"],
                port=self.config["port"],
                password=self.config["password"],
                db=self.config["db"],
                decode_responses=True,
                socket_connect_timeout=2,   # 2秒连接超时，快速失败
                socket_timeout=1
            )
            self._client.ping()
            self._available = True
            print(f"[Redis] 连接成功 {self.config['host']}:{self.config['port']}")
        except Exception as e:
            self._available = False
            print(f"[Redis] 连接失败（将跳过缓存）: {e}")

    # ── 推荐结果缓存 ────────────────────────────────────────────

    def get_recommendations(
        self,
        user_id: int,
        top_k: int,
        mode: str = "hybrid"
    ) -> list | None:
        """
        读取缓存的推荐结果

        Returns:
            推荐列表（如果缓存命中），否则返回 None
        """
        if not self._available:
            return None
        try:
            key = self._rec_key(user_id, top_k, mode)
            data = self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception:
            return None

    def set_recommendations(
        self,
        user_id: int,
        top_k: int,
        recommendations: list,
        mode: str = "hybrid",
        n_user_interactions: int = 0
    ) -> None:
        """
        写入推荐结果缓存

        TTL 策略：活跃用户行为多，推荐结果更新快，缓存时间短
        """
        if not self._available:
            return
        try:
            key = self._rec_key(user_id, top_k, mode)
            # 活跃用户缓存更短（行为变化快，推荐结果需要更新）
            ttl = TTL_ACTIVE_USER if n_user_interactions >= 20 else TTL_NORMAL_USER

            class _NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.integer):
                        return int(obj)
                    return super().default(obj)

            self._client.setex(key, ttl, json.dumps(recommendations, cls=_NumpyEncoder))
        except Exception:
            pass  # 缓存写入失败不影响主流程

    # ── 相似电影缓存 ────────────────────────────────────────────

    def get_similar(self, movie_id: int, top_k: int) -> list | None:
        if not self._available:
            return None
        try:
            key = f"{KEY_PREFIX}:similar:{movie_id}:{top_k}"
            data = self._client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None

    def set_similar(self, movie_id: int, top_k: int, result: list) -> None:
        if not self._available:
            return
        try:
            key = f"{KEY_PREFIX}:similar:{movie_id}:{top_k}"
            self._client.setex(key, TTL_SIMILAR, json.dumps(result))
        except Exception:
            pass

    # ── 缓存管理 ────────────────────────────────────────────────

    def invalidate_user(self, user_id: int) -> int:
        """
        清除某个用户的所有缓存
        场景：用户产生新的评分行为后，旧的推荐结果需要失效
        """
        if not self._available:
            return 0
        try:
            pattern = f"{KEY_PREFIX}:{user_id}:*"
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception:
            return 0

    def invalidate_all(self) -> int:
        """
        清除所有推荐缓存
        场景：模型重训完成后，所有旧缓存都要失效
        """
        if not self._available:
            return 0
        try:
            keys = self._client.keys(f"{KEY_PREFIX}:*")
            if keys:
                count = self._client.delete(*keys)
                print(f"[Redis] 清除 {count} 个缓存 key")
                return count
            return 0
        except Exception:
            return 0

    def cache_stats(self) -> dict:
        """查看缓存统计信息"""
        if not self._available:
            return {"available": False}
        try:
            info = self._client.info("stats")
            rec_keys = len(self._client.keys(f"{KEY_PREFIX}:*"))
            return {
                "available": True,
                "cached_keys": rec_keys,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except Exception:
            return {"available": False}

    @staticmethod
    def _rec_key(user_id: int, top_k: int, mode: str) -> str:
        return f"{KEY_PREFIX}:{user_id}:{top_k}:{mode}"


# 全局单例（整个应用共享一个连接）
_cache_instance: RecommendationCache | None = None


def get_cache() -> RecommendationCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RecommendationCache()
    return _cache_instance
