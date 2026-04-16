"""
models/hybrid.py
混合推荐模型：协同过滤 + 内容过滤加权融合

面试要点：
1. 为什么要混合？→ 优缺点互补
   - 协同过滤：个性化强，但冷启动差
   - 内容过滤：冷启动好，但缺乏惊喜
2. 权重如何确定？→ 基于用户行为数量动态调整
   - 新用户历史少 → 加大内容过滤权重
   - 老用户历史多 → 加大协同过滤权重
3. 生产中更复杂：学习排序（LTR）替代手动加权
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix

from .collaborative import SVDRecommender
from .content_based import ContentBasedRecommender


class HybridRecommender:
    """
    加权混合推荐器

    权重策略：动态权重（基于用户行为数量）
    - 新用户（< 5 条历史）：CF权重=0.2，CB权重=0.8
    - 普通用户（5-20条）：CF权重=0.5，CB权重=0.5
    - 活跃用户（>20条）：CF权重=0.8，CB权重=0.2
    """

    def __init__(
        self,
        cf_model: SVDRecommender,
        cb_model: ContentBasedRecommender,
        cf_weight: float = None   # None 表示使用动态权重
    ):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self._static_cf_weight = cf_weight

    def _get_cf_weight(self, n_interactions: int) -> float:
        """动态权重：根据用户历史行为数量调整"""
        if self._static_cf_weight is not None:
            return self._static_cf_weight
        if n_interactions < 5:
            return 0.2   # 冷启动用户，信任内容过滤
        elif n_interactions < 20:
            return 0.5   # 普通用户，均衡
        else:
            return 0.8   # 活跃用户，信任协同过滤

    def recommend(
        self,
        user_idx: int,
        interaction_matrix: csr_matrix,
        item2idx: dict,
        idx2item: dict,
        top_k: int = 10
    ) -> list[dict]:
        """
        混合推荐：融合 CF 和 CB 分数

        实现步骤：
        1. 分别获取 CF 和 CB 的候选集（各取 2*top_k）
        2. 归一化两者分数到 [0, 1]
        3. 加权融合
        4. 排序取 Top-K

        Returns:
            [{"item_idx": int, "score": float, "source": str}, ...]
        """
        # 用户历史行为数量
        n_interactions = interaction_matrix[user_idx].nnz
        cf_weight = self._get_cf_weight(n_interactions)
        cb_weight = 1.0 - cf_weight

        n_items = interaction_matrix.shape[1]
        candidate_k = min(top_k * 3, n_items - 1)

        # ── 协同过滤推荐 ──────────────────────────────────────────
        cf_results = {}
        try:
            cf_recs = self.cf_model.recommend(
                user_idx, interaction_matrix, top_k=candidate_k
            )
            if cf_recs:
                cf_scores = np.array([s for _, s in cf_recs])
                cf_scores_norm = self._minmax_normalize(cf_scores)
                for (item_idx, _), norm_score in zip(cf_recs, cf_scores_norm):
                    cf_results[item_idx] = norm_score
        except Exception as e:
            print(f"[混合] CF 推荐失败: {e}")

        # ── 内容过滤推荐 ──────────────────────────────────────────
        cb_results = {}
        try:
            cb_recs = self.cb_model.recommend_for_user(
                user_id=None,
                interaction_matrix=interaction_matrix,
                user_idx=user_idx,
                item2idx=item2idx,
                idx2item=idx2item,
                top_k=candidate_k
            )
            if cb_recs:
                # cb 返回的是 (movie_id, score)，需要转换为 item_idx
                cb_scores_raw = [s for _, s in cb_recs]
                cb_scores_norm = self._minmax_normalize(np.array(cb_scores_raw))
                for (movie_id, _), norm_score in zip(cb_recs, cb_scores_norm):
                    item_idx = item2idx.get(movie_id)
                    if item_idx is not None:
                        cb_results[item_idx] = norm_score
        except Exception as e:
            print(f"[混合] CB 推荐失败: {e}")

        # ── 融合分数 ──────────────────────────────────────────────
        all_items = set(cf_results.keys()) | set(cb_results.keys())
        fused_scores = {}

        for item_idx in all_items:
            cf_s = cf_results.get(item_idx, 0.0)
            cb_s = cb_results.get(item_idx, 0.0)
            fused_scores[item_idx] = cf_weight * cf_s + cb_weight * cb_s

        # 排序取 Top-K
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for item_idx, score in sorted_items[:top_k]:
            # 判断推荐来源（用于可解释性）
            has_cf = item_idx in cf_results
            has_cb = item_idx in cb_results
            source = "hybrid" if (has_cf and has_cb) else ("cf" if has_cf else "cb")
            results.append({
                "item_idx": item_idx,
                "score": round(score, 4),
                "source": source,
                "cf_score": round(cf_results.get(item_idx, 0.0), 4),
                "cb_score": round(cb_results.get(item_idx, 0.0), 4)
            })

        return results

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        # 先把 NaN 替换成 0
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-8:
            return np.ones_like(scores)
        return (scores - s_min) / (s_max - s_min)

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.data_loader import generate_mock_data, DataProcessor

    ratings, movies = generate_mock_data(n_users=500, n_movies=30)
    processor = DataProcessor()
    train_df, test_df = processor.fit_transform(ratings)
    matrix = processor.build_interaction_matrix(train_df)

    cf = SVDRecommender(n_factors=20).fit(matrix)
    cb = ContentBasedRecommender().fit(movies)
    hybrid = HybridRecommender(cf_model=cf, cb_model=cb)

    results = hybrid.recommend(
        user_idx=0,
        interaction_matrix=matrix,
        item2idx=processor.item2idx,
        idx2item=processor.idx2item,
        top_k=5
    )

    print("\n[混合推荐] 用户0 Top-5:")
    for r in results:
        movie_id = processor.idx2item[r["item_idx"]]
        title = movies[movies.movie_id == movie_id].title.values[0]
        print(f"  {title}: 融合分={r['score']:.3f} "
              f"(CF={r['cf_score']:.3f}, CB={r['cb_score']:.3f}) [{r['source']}]")
