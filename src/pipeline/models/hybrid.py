"""Hybrid recommender combining collaborative/sequential models and content embeddings."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pipeline.models.base import Recommender
from pipeline.models.embedding import ItemIndex
from pipeline.ranking.diversity import maximal_marginal_relevance

logger = logging.getLogger(__name__)


class HybridRecommender(Recommender):
    """Hybrid recommender blending collaborative filtering with content embeddings.

    Solves both user and item cold start:
    - User cold start (< cold_start_threshold interactions): surfaces popularity
      or onboarding-genre recommendations.
    - Item cold start (items absent from collaborative history): surfaces items via
      content embeddings and cosine similarity against the user's vector.
    """

    def __init__(
        self,
        collaborative_model: Any,
        content_index: ItemIndex | None = None,
        popular_items: list[int] | None = None,
        movie_lookup: dict[int, dict] | None = None,
        alpha: float = 0.6,
        cold_start_threshold: int = 3,
    ) -> None:
        self.collaborative_model = collaborative_model
        self.content_index = content_index
        self.popular_items = popular_items or []
        self.movie_lookup = movie_lookup or {}
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.cold_start_threshold = cold_start_threshold

    def fit(self, train_data: Any) -> None:
        if hasattr(self.collaborative_model, "fit"):
            self.collaborative_model.fit(train_data)

    def _recommend_cold_start_user(
        self,
        k: int,
        onboarding_genres: list[str] | None = None,
        exclude: set[int] | None = None,
    ) -> list[int]:
        exclude_set = exclude or set()

        if onboarding_genres:
            target_genres = {g.strip().lower() for g in onboarding_genres if g.strip()}
            matching: list[int] = []
            for iid in self.popular_items:
                if iid in exclude_set:
                    continue
                info = self.movie_lookup.get(iid, {})
                item_genres = {
                    g.strip().lower()
                    for g in str(info.get("genres", "")).replace("|", ",").split(",")
                    if g.strip()
                }
                if item_genres & target_genres:
                    matching.append(iid)
                if len(matching) >= k:
                    break
            if matching:
                return matching

        # Fallback to general popular items
        return [i for i in self.popular_items if i not in exclude_set][:k]

    def recommend(
        self,
        history: list[int],
        k: int = 10,
        onboarding_genres: list[str] | None = None,
        diversity: float = 0.0,
        exclude: set[int] | None = None,
    ) -> list[int]:
        recs_with_scores = self.recommend_with_scores(
            history=history,
            k=k,
            onboarding_genres=onboarding_genres,
            diversity=diversity,
            exclude=exclude,
        )
        return [iid for iid, _ in recs_with_scores]

    def recommend_with_scores(
        self,
        history: list[int],
        k: int = 10,
        onboarding_genres: list[str] | None = None,
        diversity: float = 0.0,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        exclude_set = (set(history) if history else set()) | (exclude or set())

        # 1. Cold start user: fewer than threshold interactions
        if len(history) < self.cold_start_threshold:
            cold_ids = self._recommend_cold_start_user(
                k=k, onboarding_genres=onboarding_genres, exclude=exclude_set
            )
            return [(iid, 1.0 - (idx * 0.05)) for idx, iid in enumerate(cold_ids)]

        fetch_k = max(k * 4, 40)

        # 2. Collaborative candidates
        collab_scores: dict[int, float] = {}
        if hasattr(self.collaborative_model, "predict_next_items"):
            collab_ids = self.collaborative_model.predict_next_items(
                self.collaborative_model, history, top_k=fetch_k, exclude=exclude_set
            )
            for rank, iid in enumerate(collab_ids):
                collab_scores[iid] = 1.0 / (rank + 1.0)
        elif hasattr(self.collaborative_model, "recommend"):
            collab_ids = self.collaborative_model.recommend(history, k=fetch_k)
            for rank, iid in enumerate(collab_ids):
                if iid not in exclude_set:
                    collab_scores[iid] = 1.0 / (rank + 1.0)

        # 3. Content candidates (includes cold items with 0 interactions)
        content_scores: dict[int, float] = {}
        if self.content_index is not None:
            content_res = self.content_index.search(history, top_k=fetch_k, exclude=exclude_set)
            for iid, sc in content_res:
                content_scores[iid] = sc

        all_candidate_ids = list(set(collab_scores.keys()) | set(content_scores.keys()))
        if not all_candidate_ids:
            fallback = [i for i in self.popular_items if i not in exclude_set][:k]
            return [(i, 0.5) for i in fallback]

        # Normalize collab scores to [0, 1]
        c_vals = list(collab_scores.values())
        max_c = max(c_vals) if c_vals else 1.0
        norm_collab = {i: sc / max_c for i, sc in collab_scores.items()}

        # Normalize content scores to [0, 1]
        cnt_vals = list(content_scores.values())
        min_cnt = min(cnt_vals) if cnt_vals else 0.0
        max_cnt = max(cnt_vals) if cnt_vals else 1.0
        span_cnt = max_cnt - min_cnt
        norm_content = {
            i: (sc - min_cnt) / span_cnt if span_cnt > 1e-9 else 0.5
            for i, sc in content_scores.items()
        }

        # 4. Blend scores: alpha * collab + (1 - alpha) * content
        hybrid_scores: dict[int, float] = {}
        for iid in all_candidate_ids:
            sc_c = norm_collab.get(iid, 0.0)
            sc_cnt = norm_content.get(iid, 0.0)
            if iid in norm_collab and iid in norm_content:
                hybrid_scores[iid] = (self.alpha * sc_c) + ((1.0 - self.alpha) * sc_cnt)
            elif iid in norm_content:
                # Cold item: scored purely via content embeddings
                hybrid_scores[iid] = (1.0 - self.alpha) * sc_cnt
            else:
                hybrid_scores[iid] = self.alpha * sc_c

        # 5. Diversity re-ranking via MMR if requested
        if diversity > 0.0 and self.content_index is not None:
            feature_map: dict[int, Any] = {}
            for iid in all_candidate_ids:
                if iid in self.content_index.id_to_idx:
                    idx = self.content_index.id_to_idx[iid]
                    feature_map[iid] = self.content_index.embeddings[idx]
                else:
                    info = self.movie_lookup.get(iid, {})
                    feature_map[iid] = set(str(info.get("genres", "")).split("|"))

            ranked_ids = maximal_marginal_relevance(
                all_candidate_ids,
                item_features=feature_map,
                scores=hybrid_scores,
                diversity=diversity,
                k=k,
            )
        else:
            ranked_ids = sorted(
                all_candidate_ids, key=lambda iid: hybrid_scores.get(iid, 0.0), reverse=True
            )[:k]

        return [(iid, hybrid_scores.get(iid, 0.0)) for iid in ranked_ids]
