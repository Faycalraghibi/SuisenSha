"""Two-Stage Recommender Architecture: Candidate Retrieval + Deep Re-Ranking.

Stage 1: Fast Candidate Retrieval (Two-Tower FAISS index or embedding ANN) -> Top M candidates.
Stage 2: Deep Re-Ranking (SASRec transformer or neural scoring) -> Top K recommendations.
"""

from __future__ import annotations

import logging
from typing import Any

from pipeline.models.base import Recommender
from pipeline.models.sequential import SASRec
from pipeline.models.two_tower import TwoTowerModel
from pipeline.ranking.diversity import maximal_marginal_relevance
from pipeline.ranking.rules import RuleEngine

logger = logging.getLogger(__name__)


class TwoStageRecommender(Recommender):
    """Two-stage pipeline: Fast ANN Candidate Retrieval followed by Deep Sequential Re-Ranking."""

    def __init__(
        self,
        retrieval_model: TwoTowerModel,
        reranker_model: SASRec | Any,
        candidate_pool_size: int = 50,
        rule_engine: RuleEngine | None = None,
        diversity_weight: float = 0.0,
    ) -> None:
        self.retrieval_model = retrieval_model
        self.reranker_model = reranker_model
        self.candidate_pool_size = candidate_pool_size
        self.rule_engine = rule_engine or RuleEngine()
        self.diversity_weight = diversity_weight

    def fit(self, train_data: Any) -> None:
        """Fit both retrieval and re-ranking stages."""
        if hasattr(self.retrieval_model, "fit"):
            self.retrieval_model.fit(train_data)
        if hasattr(self.reranker_model, "fit"):
            self.reranker_model.fit(train_data)

    def recommend(
        self,
        history: list[int],
        k: int = 10,
        tenant_id: str = "default",
        item_features: dict[int, set[str]] | None = None,
    ) -> list[int]:
        """Execute two-stage recommendation workflow.

        1. Stage 1: Retrieve candidate_pool_size items via fast retrieval.
        2. Stage 2: Re-rank retrieved candidates via deep scoring.
        3. Optional: MMR diversity and post-ranking business rules.
        """
        # Stage 1: Candidate Retrieval
        candidates_with_scores = self.retrieval_model.retrieve_candidates(
            history, top_k=self.candidate_pool_size
        )
        if not candidates_with_scores:
            return []

        cand_ids = [cid for cid, _ in candidates_with_scores]

        # Stage 2: Re-Ranking
        if hasattr(self.reranker_model, "last_logits"):
            import torch

            device = next(self.reranker_model.parameters()).device
            seq_tensor = torch.tensor([history[-50:]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.reranker_model.last_logits(seq_tensor).squeeze(0)
                scored_cands: list[tuple[int, float]] = []
                for cid in cand_ids:
                    if cid < logits.size(0):
                        scored_cands.append((cid, float(logits[cid].item())))
                    else:
                        scored_cands.append((cid, 0.0))
                # Sort descending by re-ranker score
                scored_cands.sort(key=lambda x: x[1], reverse=True)
                reranked_ids = [cid for cid, _ in scored_cands]
                score_dict = dict(scored_cands)
        elif hasattr(self.reranker_model, "recommend"):
            # Fallback to model's own recommendation ordering over candidate subset
            full_ranked = self.reranker_model.recommend(history, k=len(cand_ids) * 2)
            ranked_set = set(cand_ids)
            reranked_ids = [cid for cid in full_ranked if cid in ranked_set]
            # Append any unranked candidates
            remaining = [cid for cid in cand_ids if cid not in set(reranked_ids)]
            reranked_ids.extend(remaining)
            score_dict = {cid: 1.0 / (idx + 1.0) for idx, cid in enumerate(reranked_ids)}
        else:
            reranked_ids = cand_ids
            score_dict = dict(candidates_with_scores)

        # Stage 2.5: Maximal Marginal Relevance Diversity (if enabled)
        if self.diversity_weight > 0.0 and item_features:
            reranked_ids = maximal_marginal_relevance(
                candidate_ids=reranked_ids,
                item_features=item_features,
                scores=score_dict,
                diversity=self.diversity_weight,
                k=k,
            )

        # Stage 3: Business rules application
        if self.rule_engine:
            cand_dicts = [
                {"item_id": cid, "score": score_dict.get(cid, 0.0)} for cid in reranked_ids
            ]
            ranked_dicts = self.rule_engine.apply(cand_dicts, top_k=k, tenant_id=tenant_id)
            return [int(it["item_id"]) for it in ranked_dicts[:k]]

        return reranked_ids[:k]
