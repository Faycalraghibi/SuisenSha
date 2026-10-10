from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.config import SEED
from pipeline.models.base import Recommender

logger = logging.getLogger(__name__)


class RandomRecommender(Recommender):
    """Recommends random unvisited items from the training catalog."""

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self.all_items: list[int] = []
        self._rng = random.Random(seed)

    def fit(self, train_data: Any) -> None:
        if isinstance(train_data, dict):
            items = set()
            for seq in train_data.values():
                items.update(seq)
            self.all_items = sorted(items)
        elif isinstance(train_data, list | set):
            self.all_items = sorted(train_data)
        logger.info("RandomRecommender fitted with %d items", len(self.all_items))

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        seen = set(history)
        candidates = [i for i in self.all_items if i not in seen]
        if not candidates:
            return []
        sample_k = min(k, len(candidates))
        return self._rng.sample(candidates, sample_k)


class PopularityRecommender(Recommender):
    """Recommends globally most popular items not yet seen by the user."""

    def __init__(self) -> None:
        self.popular_items: list[int] = []

    def fit(self, train_data: Any) -> None:
        counter: Counter[int] = Counter()
        if isinstance(train_data, dict):
            for seq in train_data.values():
                counter.update(seq)
        self.popular_items = [item for item, _ in counter.most_common()]
        logger.info("PopularityRecommender fitted with %d unique items", len(self.popular_items))

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        seen = set(history)
        recs = [item for item in self.popular_items if item not in seen]
        return recs[:k]


class ItemKNNRecommender(Recommender):
    """Item-item collaborative filtering using cosine similarity over co-occurrence."""

    def __init__(self, k_neighbors: int = 20) -> None:
        self.k_neighbors = k_neighbors
        self.similarity: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self.popular_items: list[int] = []

    def fit(self, train_data: Any) -> None:
        # Build item co-occurrence counts
        item_counts: Counter[int] = Counter()
        co_counts: dict[int, Counter[int]] = defaultdict(Counter)

        if isinstance(train_data, dict):
            for seq in train_data.values():
                unique_items = set(seq)
                for item in unique_items:
                    item_counts[item] += 1
                for i1 in unique_items:
                    for i2 in unique_items:
                        if i1 != i2:
                            co_counts[i1][i2] += 1

        self.popular_items = [i for i, _ in item_counts.most_common()]

        # Compute cosine similarity: co_counts[i1][i2] / sqrt(count(i1) * count(i2))
        for i1, neighbors in co_counts.items():
            sims: list[tuple[int, float]] = []
            for i2, co_cnt in neighbors.items():
                denom = np.sqrt(item_counts[i1] * item_counts[i2])
                sim = float(co_cnt / denom) if denom > 0 else 0.0
                sims.append((i2, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            self.similarity[i1] = sims[: self.k_neighbors]

        logger.info("ItemKNNRecommender fitted for %d items", len(self.similarity))

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        seen = set(history)
        scores: dict[int, float] = defaultdict(float)

        for item in history:
            for neighbor, sim in self.similarity.get(item, []):
                if neighbor not in seen:
                    scores[neighbor] += sim

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recs = [item for item, _ in sorted_candidates[:k]]
        if len(recs) < k:
            for pop_item in self.popular_items:
                if pop_item not in seen and pop_item not in scores:
                    recs.append(pop_item)
                    if len(recs) >= k:
                        break
        return recs[:k]


class _BPRDataset(Dataset):
    def __init__(
        self,
        interactions: list[tuple[int, int]],
        num_items: int,
        num_negatives: int = 1,
    ) -> None:
        self.interactions = interactions
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_items: dict[int, set[int]] = defaultdict(set)
        for u, i in interactions:
            self.user_items[u].add(i)

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        u, pos_i = self.interactions[idx]
        neg_i = random.randint(1, self.num_items)
        while neg_i in self.user_items[u]:
            neg_i = random.randint(1, self.num_items)
        return u, pos_i, neg_i


class _BPRModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, embedding_dim)
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        u_emb = self.user_emb(users)
        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)

        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        # BPR loss: -log(sigmoid(pos_scores - neg_scores))
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss

    def predict_all(self, user_idx: int) -> torch.Tensor:
        u_emb = self.user_emb.weight[user_idx]  # (dim,)
        all_items = self.item_emb.weight  # (num_items + 1, dim)
        return torch.matmul(all_items, u_emb)  # (num_items + 1,)


class BPRMFRecommender(Recommender):
    """Bayesian Personalized Ranking Matrix Factorization."""

    def __init__(
        self,
        embedding_dim: int = 32,
        lr: float = 0.01,
        epochs: int = 5,
        batch_size: int = 256,
        seed: int = SEED,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.num_items = 0
        self.popular_items: list[int] = []
        self.model: _BPRModel | None = None
        self.user_map: dict[int, int] = {}

    def fit(self, train_data: Any) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        interactions: list[tuple[int, int]] = []
        user_counter: Counter[int] = Counter()
        item_counter: Counter[int] = Counter()

        if isinstance(train_data, dict):
            for uid, seq in train_data.items():
                if uid not in self.user_map:
                    self.user_map[uid] = len(self.user_map) + 1
                u_idx = self.user_map[uid]
                for item in seq:
                    interactions.append((u_idx, item))
                    item_counter[item] += 1
                user_counter[u_idx] += len(seq)

        self.popular_items = [i for i, _ in item_counter.most_common()]
        self.num_items = max(item_counter.keys()) if item_counter else 0
        num_users = len(self.user_map)

        if not interactions or self.num_items == 0:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _BPRModel(num_users, self.num_items, self.embedding_dim).to(device)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = _BPRDataset(interactions, self.num_items)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for u, pos, neg in loader:
                loss = self.model(u.to(device), pos.to(device), neg.to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()

        self.model.eval()
        logger.info("BPRMFRecommender fitted (%d users, %d items)", num_users, self.num_items)

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        seen = set(history)
        if self.model is None or not history:
            return [i for i in self.popular_items if i not in seen][:k]

        device = next(self.model.parameters()).device
        # Find active items and construct pseudo-user profile as average of history embeddings
        valid_history = [i for i in history if 0 < i <= self.num_items]
        if not valid_history:
            return [i for i in self.popular_items if i not in seen][:k]

        with torch.no_grad():
            hist_tensor = torch.tensor(valid_history, dtype=torch.long, device=device)
            item_embs = self.model.item_emb(hist_tensor)  # (len, dim)
            user_vec = item_embs.mean(dim=0)  # (dim,)
            scores = torch.matmul(self.model.item_emb.weight, user_vec)  # (num_items + 1,)
            scores[0] = -float("inf")
            for i in seen:
                if 0 <= i < scores.shape[0]:
                    scores[i] = -float("inf")
            top_indices = scores.topk(min(k, self.num_items)).indices.cpu().tolist()

        return top_indices
