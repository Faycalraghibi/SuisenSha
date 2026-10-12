"""Two-Tower Candidate Retrieval Model with FAISS Integration.

Two separate neural networks:
- User Tower: projects user sequential history into a d-dimensional vector.
- Item Tower: projects catalog item embeddings into a d-dimensional vector.
"""

from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

from pipeline.config import SEED
from pipeline.models.base import Recommender

logger = logging.getLogger(__name__)


class TwoTowerDataset(Dataset):
    """Training pairs (user_history, target_item) for two-tower retrieval."""

    def __init__(
        self,
        sequences: dict[int, list[int]],
        max_seq_len: int = 50,
    ) -> None:
        self.samples: list[tuple[list[int], int]] = []
        self.max_seq_len = max_seq_len

        for seq in sequences.values():
            if len(seq) < 2:
                continue
            for idx in range(1, len(seq)):
                sub_seq = seq[max(0, idx - max_seq_len) : idx]
                target = seq[idx]
                self.samples.append((sub_seq, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sub_seq, target = self.samples[idx]
        padded = sub_seq + [0] * (self.max_seq_len - len(sub_seq))
        return torch.tensor(padded, dtype=torch.long), target


class UserTower(nn.Module):
    """User tower encoding interaction history into a normalized representation."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Forward pass pooling historical item embeddings into user representation.

        Args:
            history: (batch_size, seq_len) item IDs

        Returns:
            user_emb: (batch_size, output_dim) L2-normalized vector
        """
        # Masked average pooling over valid items
        mask = (history != 0).unsqueeze(-1).float()  # (B, L, 1)
        emb = self.item_embedding(history) * mask  # (B, L, D)
        sum_emb = emb.sum(dim=1)  # (B, D)
        lens = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        mean_emb = sum_emb / lens

        proj = self.mlp(mean_emb)
        return F.normalize(proj, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item tower projecting catalog item IDs into a normalized representation."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass computing normalized item vectors."""
        emb = self.item_embedding(item_ids)
        proj = self.mlp(emb)
        return F.normalize(proj, p=2, dim=-1)


class TwoTowerModel(nn.Module, Recommender):
    """Two-Tower Candidate Retrieval with sub-millisecond FAISS search."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        output_dim: int = 64,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.output_dim = output_dim
        self.temperature = temperature

        self.user_tower = UserTower(num_items, embedding_dim, output_dim)
        self.item_tower = ItemTower(num_items, embedding_dim, output_dim)

        self._faiss_index: faiss.Index | None = None
        self._index_item_ids: np.ndarray | None = None

    def forward(
        self,
        user_history: torch.Tensor,
        target_items: torch.Tensor,
    ) -> torch.Tensor:
        """Compute in-batch sampled softmax loss.

        Cosine similarities between user representations and all items in batch.
        """
        user_vecs = self.user_tower(user_history)  # (B, D)
        item_vecs = self.item_tower(target_items)  # (B, D)

        # In-batch similarity matrix
        sim_matrix = torch.matmul(user_vecs, item_vecs.t()) / self.temperature  # (B, B)
        targets = torch.arange(user_vecs.size(0), device=user_vecs.device)
        loss = F.cross_entropy(sim_matrix, targets)
        return loss

    def build_faiss_index(self) -> faiss.Index:
        """Build and populate FAISS Inner Product index over all catalog items."""
        self.eval()
        device = next(self.parameters()).device
        all_ids = torch.arange(1, self.num_items + 1, device=device)

        with torch.no_grad():
            all_vecs = self.item_tower(all_ids).cpu().numpy().astype(np.float32)

        index = faiss.IndexFlatIP(self.output_dim)
        index.add(all_vecs)
        self._faiss_index = index
        self._index_item_ids = np.arange(1, self.num_items + 1)
        return index

    def retrieve_candidates(self, history: list[int], top_k: int = 50) -> list[tuple[int, float]]:
        if self._faiss_index is None or self._index_item_ids is None:
            self.build_faiss_index()

        assert self._faiss_index is not None
        assert self._index_item_ids is not None

        self.eval()

        device = next(self.parameters()).device
        valid_history = [iid for iid in history if 0 < iid <= self.num_items]
        if not valid_history:
            return [(idx, 1.0) for idx in range(1, min(top_k + 1, self.num_items + 1))]

        hist_tensor = torch.tensor([valid_history], dtype=torch.long, device=device)
        with torch.no_grad():
            u_vec = self.user_tower(hist_tensor).cpu().numpy().astype(np.float32)

        # Exclude historical items
        search_k = min(top_k + len(valid_history) + 5, self.num_items)
        scores, indices = self._faiss_index.search(u_vec, search_k)

        candidates: list[tuple[int, float]] = []
        seen = set(valid_history)
        for score, idx in zip(scores[0], indices[0], strict=False):
            item_id = int(self._index_item_ids[idx])
            if item_id not in seen:
                candidates.append((item_id, float(score)))
                if len(candidates) >= top_k:
                    break
        return candidates

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        """Recommender protocol interface."""
        candidates = self.retrieve_candidates(history, top_k=k)
        return [cid for cid, _ in candidates]

    def fit(self, train_data: Any) -> None:
        """Fit model parameters on sequential interaction history."""
        if isinstance(train_data, dict):
            train_two_tower(train_data, model=self)


def train_two_tower(
    sequences: dict[int, list[int]],
    model: TwoTowerModel | None = None,
    num_items: int | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    embedding_dim: int = 64,
    output_dim: int = 64,
    seed: int = SEED,
) -> TwoTowerModel:
    """Train two-tower candidate retrieval model with in-batch negatives."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if num_items is None:
        num_items = max(max(seq) for seq in sequences.values() if seq)

    if model is None:
        model = TwoTowerModel(
            num_items=num_items,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
        )

    dataset = TwoTowerDataset(sequences)
    if len(dataset) == 0:
        model.build_faiss_index()
        return model

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for histories, targets in dataloader:
            if histories.size(0) <= 1:
                continue
            optimizer.zero_grad()
            loss = model(histories, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    model.build_faiss_index()
    return model
