"""
ab_test/ab_manager.py
A/B 测试框架

核心思想：
- 每个用户根据 user_id 哈希稳定分配到某个实验组
- 记录每次推荐曝光（impression）和用户点击（click）
- 统计各组 CTR（点击率），用 Z-test 判断是否有统计显著差异

面试要点：
1. 为什么用哈希分组而不是随机？→ 保证同一用户每次进同一组（稳定性）
2. 什么是 AA 测试？→ 上线前验证分组是否均匀，两组结果应无差异
3. 什么时候结束实验？→ 达到显著性（p < 0.05）且样本量足够
4. 辛普森悖论怎么处理？→ 按用户活跃度分层分析
"""
from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime


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


class ABManager:
    """
    A/B 测试管理器

    使用方式：
        manager = ABManager()
        variant = manager.get_variant(user_id, experiment_id=1)
        # variant 是 "cf" / "hybrid" / "cb" 之一

        manager.log_impression(user_id, experiment_id=1, variant=variant)
        manager.log_click(user_id, experiment_id=1, variant=variant, movie_id=123)
    """

    def get_variant(self, user_id: int, experiment_id: int = 1) -> str:
        """
        根据 user_id 哈希稳定分配实验组

        原理：
        - MD5(user_id + experiment_id) → 取前8位 → 转10进制 → 模 N
        - 同一用户在同一实验中永远得到同一组（不随机漂移）
        - 不同实验用不同 salt，避免用户在所有实验都在同一组
        """
        try:
            conn = get_db()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT variants FROM ab_experiments WHERE id=%s AND status='active'",
                    (experiment_id,)
                )
                row = cursor.fetchone()
            conn.close()

            if not row:
                return "hybrid"  # 实验不存在时的默认值

            import json
            variants = json.loads(row["variants"]) if isinstance(row["variants"], str) else row["variants"]

            # 哈希分组
            key = f"{user_id}_{experiment_id}"
            hash_val = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            idx = hash_val % len(variants)
            return variants[idx]

        except Exception:
            return "hybrid"

    def log_impression(
        self,
        user_id: int,
        experiment_id: int,
        variant: str
    ) -> None:
        """
        记录推荐曝光事件
        场景：用户获取到推荐列表时调用
        """
        self._log_event(user_id, experiment_id, variant, "impression", None)

    def log_click(
        self,
        user_id: int,
        experiment_id: int,
        variant: str,
        movie_id: int
    ) -> None:
        """
        记录点击事件
        场景：用户点击推荐列表中的某部电影时调用
        """
        self._log_event(user_id, experiment_id, variant, "click", movie_id)

    def _log_event(
        self,
        user_id: int,
        experiment_id: int,
        variant: str,
        event_type: str,
        movie_id: int | None
    ) -> None:
        try:
            conn = get_db()
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ab_events
                        (experiment_id, user_id, variant, event_type, movie_id)
                    VALUES (%s, %s, %s, %s, %s)
                """, (experiment_id, user_id, variant, event_type, movie_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[A/B] 记录事件失败: {e}")

    def get_stats(self, experiment_id: int = 1) -> dict:
        """
        获取实验统计结果

        返回每个实验组的：
        - impressions：曝光次数
        - clicks：点击次数
        - ctr：点击率 = clicks / impressions
        - p_value：与对照组的统计显著性（Z-test）

        面试：
        - CTR 提升 1% 意味着什么？→ 取决于基线，要看绝对值和相对提升
        - p_value < 0.05 代表什么？→ 差异非随机的概率 > 95%
        - 实验跑多久？→ 至少覆盖一个完整用户行为周期（通常 1~2 周）
        """
        try:
            conn = get_db()
            with conn.cursor() as cursor:
                # 各组曝光数
                cursor.execute("""
                    SELECT variant,
                           SUM(event_type='impression') AS impressions,
                           SUM(event_type='click')      AS clicks
                    FROM ab_events
                    WHERE experiment_id=%s
                    GROUP BY variant
                """, (experiment_id,))
                rows = cursor.fetchall()

                # 实验基本信息
                cursor.execute(
                    "SELECT * FROM ab_experiments WHERE id=%s",
                    (experiment_id,)
                )
                exp = cursor.fetchone()
            conn.close()

            stats = {}
            for row in rows:
                imp = int(row["impressions"] or 0)
                clk = int(row["clicks"] or 0)
                ctr = clk / imp if imp > 0 else 0.0
                stats[row["variant"]] = {
                    "impressions": imp,
                    "clicks":      clk,
                    "ctr":         round(ctr * 100, 2),  # 转为百分比
                }

            # 计算 Z-test 显著性（以第一个组为对照组）
            variants = list(stats.keys())
            if len(variants) >= 2:
                control = variants[0]
                for treatment in variants[1:]:
                    p = self._z_test(
                        stats[control]["clicks"],
                        stats[control]["impressions"],
                        stats[treatment]["clicks"],
                        stats[treatment]["impressions"],
                    )
                    stats[treatment]["p_value"] = round(p, 4)
                    stats[treatment]["significant"] = p < 0.05
                stats[control]["p_value"] = None
                stats[control]["significant"] = None

            return {
                "experiment_id":   experiment_id,
                "experiment_name": exp["name"] if exp else "",
                "description":     exp["description"] if exp else "",
                "stats":           stats,
                "total_events":    sum(v["impressions"] + v["clicks"] for v in stats.values()),
            }

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _z_test(
        clicks_a: int, impressions_a: int,
        clicks_b: int, impressions_b: int
    ) -> float:
        """
        双比例 Z 检验
        H0: CTR_A == CTR_B
        返回 p-value（双尾）

        公式：
        p_a = clicks_a / impressions_a
        p_b = clicks_b / impressions_b
        p_pool = (clicks_a + clicks_b) / (impressions_a + impressions_b)
        SE = sqrt(p_pool * (1-p_pool) * (1/n_a + 1/n_b))
        z = (p_a - p_b) / SE
        p_value = 2 * (1 - Φ(|z|))
        """
        if impressions_a == 0 or impressions_b == 0:
            return 1.0

        p_a = clicks_a / impressions_a
        p_b = clicks_b / impressions_b
        p_pool = (clicks_a + clicks_b) / (impressions_a + impressions_b)

        se = math.sqrt(p_pool * (1 - p_pool) * (1/impressions_a + 1/impressions_b))
        if se == 0:
            return 1.0

        z = abs(p_a - p_b) / se

        # 正态分布近似 CDF
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        p_value = 2 * (1 - norm_cdf(z))
        return p_value
