"""LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution
    Network for Recommendation." SIGIR 2020.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.config import SEED
from pipeline.models.base import Recommender

logger = logging.getLogger(__name__)


class BPRDataset(Dataset):
    """Pairwise BPR training dataset (user, pos_item, neg_item)."""

    def __init__(
        self,
        user_items: dict[int, set[int]],
        num_items: int,
        num_negatives: int = 1,
    ) -> None:
        self.samples: list[tuple[int, int, int]] = []
        self.num_items = num_items

        for uid, items in user_items.items():
            if not items:
                continue
            for pos_item in items:
                for _ in range(num_negatives):
                    neg_item = random.randint(1, num_items)
                    while neg_item in items:
                        neg_item = random.randint(1, num_items)
                    self.samples.append((uid, pos_item, neg_item))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        return self.samples[idx]


class LightGCN(nn.Module, Recommender):
    """Light Graph Convolutional Network for Collaborative Filtering."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # User and item 0-indexed embeddings (+1 for 1-based indexing safety)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)

        self._norm_adj_matrix: torch.Tensor | None = None
        self._user_history: dict[int, set[int]] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def set_graph(self, user_items: Mapping[int, list[int] | set[int]]) -> None:
        """Construct and store the symmetrically normalized adjacency matrix."""
        self._user_history = {u: set(items) for u, items in user_items.items()}

        # Build sparse interaction matrix R
        rows: list[int] = []
        cols: list[int] = []
        for u, items in self._user_history.items():
            if u <= self.num_users:
                for i in items:
                    if i <= self.num_items:
                        rows.append(u)
                        cols.append(i)

        total_nodes = (self.num_users + 1) + (self.num_items + 1)
        if not rows:
            # Fallback identity if empty
            indices = torch.arange(total_nodes).repeat(2, 1)
            values = torch.ones(total_nodes)
            self._norm_adj_matrix = torch.sparse_coo_tensor(
                indices, values, (total_nodes, total_nodes)
            )
            return

        u_arr = np.array(rows, dtype=np.int64)
        i_arr = np.array(cols, dtype=np.int64) + (self.num_users + 1)

        # Symmetric adjacency edges
        adj_rows = np.concatenate([u_arr, i_arr])
        adj_cols = np.concatenate([i_arr, u_arr])
        adj_data = np.ones(len(adj_rows), dtype=np.float32)

        adj = sp.coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(total_nodes, total_nodes))
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
        nonzero = rowsum > 0
        d_inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)

        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

        indices = torch.tensor(np.vstack((norm_adj.row, norm_adj.col)), dtype=torch.long)
        values = torch.tensor(norm_adj.data, dtype=torch.float32)
        self._norm_adj_matrix = torch.sparse_coo_tensor(indices, values, (total_nodes, total_nodes))

    def compute_layer_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate embeddings through GCN layers and compute multi-scale mean."""
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_embeddings = [all_embeddings]

        device = all_embeddings.device
        if self._norm_adj_matrix is not None:
            adj = self._norm_adj_matrix.to(device)
            for _ in range(self.num_layers):
                all_embeddings = torch.sparse.mm(adj, all_embeddings)
                layer_embeddings.append(all_embeddings)

        # Multi-layer weighted combination (uniform weighting 1 / (K + 1))
        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        u_embed, i_embed = torch.split(final_embeddings, [self.num_users + 1, self.num_items + 1])
        return u_embed, i_embed

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute BPR loss and L2 regularization penalty for training triplets."""
        u_embed, i_embed = self.compute_layer_embeddings()

        u = u_embed[users]
        pos_i = i_embed[pos_items]
        neg_i = i_embed[neg_items]

        pos_scores = (u * pos_i).sum(dim=1)
        neg_scores = (u * neg_i).sum(dim=1)

        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # Regularization on initial (0-layer) embeddings
        reg_loss = (
            self.user_embedding(users).norm(2).pow(2)
            + self.item_embedding(pos_items).norm(2).pow(2)
            + self.item_embedding(neg_items).norm(2).pow(2)
        ) / users.size(0)

        return bpr_loss, reg_loss

    def recommend_for_user(
        self, user_id: int, k: int = 10, exclude: set[int] | None = None
    ) -> list[int]:
        """Generate top-k recommendations for a known user by ID."""
        self.eval()
        if user_id > self.num_users or user_id <= 0:
            return list(range(1, min(k + 1, self.num_items + 1)))

        with torch.no_grad():
            u_embed, i_embed = self.compute_layer_embeddings()
            u_vec = u_embed[user_id]  # (D,)
            scores = torch.mv(i_embed, u_vec)  # (num_items + 1,)

            scores[0] = -1e9
            seen = self._user_history.get(user_id, set())
            if exclude:
                seen = seen | exclude
            for iid in seen:
                if iid <= self.num_items:
                    scores[iid] = -1e9

            topk = torch.topk(scores, k=min(k, self.num_items)).indices.tolist()
            return [idx for idx in topk if idx > 0]

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        """Recommender protocol interface."""
        self.eval()
        valid_history = [iid for iid in history if 0 < iid <= self.num_items]
        if not valid_history:
            return list(range(1, min(k + 1, self.num_items + 1)))

        with torch.no_grad():
            _, i_embed = self.compute_layer_embeddings()
            # Synthesize pseudo-user vector as mean of historical item representations
            hist_tensor = torch.tensor(valid_history, dtype=torch.long, device=i_embed.device)
            u_vec = i_embed[hist_tensor].mean(dim=0)
            scores = torch.mv(i_embed, u_vec)

            scores[0] = -1e9
            for iid in valid_history:
                if iid <= self.num_items:
                    scores[iid] = -1e9

            topk = torch.topk(scores, k=min(k, self.num_items)).indices.tolist()
            return [idx for idx in topk if idx > 0]

    def fit(self, train_data: Any) -> None:
        """Fit model on user interaction history."""
        if isinstance(train_data, dict):
            train_lightgcn(train_data, model=self)


def train_lightgcn(
    sequences: dict[int, list[int]],
    model: LightGCN | None = None,
    num_users: int | None = None,
    num_items: int | None = None,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.005,
    decay: float = 1e-4,
    embedding_dim: int = 64,
    num_layers: int = 3,
    seed: int = SEED,
) -> LightGCN:
    """Train LightGCN model on bipartite user-item interactions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if num_users is None:
        num_users = max(sequences.keys()) if sequences else 0
    if num_items is None:
        num_items = max(max(seq) for seq in sequences.values() if seq)

    if model is None:
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )

    user_items = {u: set(seq) for u, seq in sequences.items()}
    model.set_graph(user_items)

    dataset = BPRDataset(user_items, num_items=num_items)
    if len(dataset) == 0:
        return model

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for users, pos_items, neg_items in dataloader:
            optimizer.zero_grad()
            bpr_loss, reg_loss = model(users, pos_items, neg_items)
            loss = bpr_loss + decay * reg_loss
            loss.backward()
            optimizer.step()

    model.eval()
    return model
