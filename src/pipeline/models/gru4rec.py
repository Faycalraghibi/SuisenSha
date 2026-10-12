"""GRU4Rec: Session and Sequential Recommendation with Recurrent Neural Networks.

Reference:
    Balázs Hidasi et al. "Session-based Recommendations with Recurrent Neural Networks."
    ICLR 2016.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.config import SEED
from pipeline.models.base import Recommender

logger = logging.getLogger(__name__)


class GRU4RecDataset(Dataset):
    """Next-item dataset for sequential training."""

    def __init__(
        self,
        sequences: dict[int, list[int]],
        max_seq_len: int = 50,
        num_items: int = 0,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        sub_seq, target = self.samples[idx]
        length = len(sub_seq)
        padded = sub_seq + [0] * (self.max_seq_len - length)
        return torch.tensor(padded, dtype=torch.long), target, length


class GRU4Rec(nn.Module, Recommender):
    """GRU-based sequential recommendation model."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        max_seq_len: int = 50,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token 0 is reserved for padding
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_items + 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, seqs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass generating prediction logits over catalog for last sequence state.

        Args:
            seqs: (batch_size, max_seq_len)
            lengths: (batch_size,) sequence lengths before padding

        Returns:
            logits: (batch_size, num_items + 1)
        """
        embedded = self.emb_dropout(self.item_embedding(seqs))  # (B, L, D)
        out, h_n = self.gru(embedded)  # out: (B, L, H), h_n: (num_layers, B, H)

        if lengths is not None:
            # Extract hidden state at actual sequence end
            idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
            last_hidden = out.gather(1, idx).squeeze(1)  # (B, H)
        else:
            # Use final layer hidden state
            last_hidden = h_n[-1]  # (B, H)

        logits = self.fc(last_hidden)  # (B, num_items + 1)
        return logits

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        """Generate top-k recommendations given user interaction history."""
        self.eval()
        device = next(self.parameters()).device
        valid_history = [iid for iid in history if 0 < iid <= self.num_items]
        if not valid_history:
            return list(range(1, min(k + 1, self.num_items + 1)))

        sub_seq = valid_history[-self.max_seq_len :]
        length = len(sub_seq)
        padded = sub_seq + [0] * (self.max_seq_len - length)

        seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
        len_tensor = torch.tensor([length], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = self.forward(seq_tensor, len_tensor).squeeze(0)  # (num_items + 1)
            # Mask padding token and previously seen items
            logits[0] = -1e9
            for iid in valid_history:
                if iid <= self.num_items:
                    logits[iid] = -1e9

            topk_indices = torch.topk(logits, k=min(k, self.num_items)).indices.tolist()
            return [idx for idx in topk_indices if idx > 0]

    def fit(self, train_data: Any) -> None:
        """Fit model parameters on training interactions / sequences."""
        if isinstance(train_data, dict):
            train_gru4rec(train_data, model=self)


def train_gru4rec(
    sequences: dict[int, list[int]],
    model: GRU4Rec | None = None,
    num_items: int | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_dim: int = 64,
    seed: int = SEED,
) -> GRU4Rec:
    """Train GRU4Rec model using cross-entropy next-item prediction loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if num_items is None:
        num_items = max(max(seq) for seq in sequences.values() if seq)

    if model is None:
        model = GRU4Rec(
            num_items=num_items,
            embedding_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

    dataset = GRU4RecDataset(sequences, max_seq_len=model.max_seq_len, num_items=num_items)
    if len(dataset) == 0:
        return model

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for seqs, targets, lengths in dataloader:
            optimizer.zero_grad()
            logits = model(seqs, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model
