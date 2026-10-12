"""BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations.

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional
    Encoder Representations from Transformer." CIKM 2019.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.config import SEED
from pipeline.models.base import Recommender

logger = logging.getLogger(__name__)


class BERT4RecDataset(Dataset):
    """Cloze (Masked Language Model style) training dataset for BERT4Rec."""

    def __init__(
        self,
        sequences: dict[int, list[int]],
        num_items: int,
        max_seq_len: int = 50,
        mask_prob: float = 0.15,
    ) -> None:
        self.samples: list[list[int]] = []
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1  # 0 is padding, 1..N items, N+1 is mask

        for seq in sequences.values():
            if len(seq) >= 2:
                self.samples.append(seq[-max_seq_len:])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx]
        masked_seq = list(seq)
        labels = [0] * len(seq)

        for i in range(len(seq)):
            if random.random() < self.mask_prob:
                labels[i] = seq[i]
                prob = random.random()
                if prob < 0.8:
                    masked_seq[i] = self.mask_token
                elif prob < 0.9:
                    # Random item replacement
                    masked_seq[i] = random.randint(1, self.num_items)
                # Remaining 10%: keep unchanged

        # Pad right to max_seq_len
        pad_len = self.max_seq_len - len(seq)
        padded_seq = masked_seq + [0] * pad_len
        padded_labels = labels + [0] * pad_len

        return (
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(padded_labels, dtype=torch.long),
        )


class BERT4Rec(nn.Module, Recommender):
    """Bidirectional Transformer for Sequential Recommendation."""

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.mask_token = num_items + 1

        # Token 0 is pad, 1..num_items are items, num_items+1 is [MASK]
        self.vocab_size = num_items + 2

        self.item_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0,
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_seq_len + 1,
            embedding_dim=hidden_dim,
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, num_items + 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        """Encode sequence with bidirectional Transformer attention."""
        seq_len = seqs.size(1)
        device = seqs.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(seqs.size(0), -1)

        x = self.item_embedding(seqs) + self.pos_embedding(positions)
        x = self.layer_norm(self.emb_dropout(x))

        # Padding mask: True where attention is forbidden (seqs == 0)
        pad_mask = seqs == 0
        hidden = self.transformer(x, src_key_padding_mask=pad_mask)
        return hidden

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """Output logits across catalog for all sequence positions: (B, L, num_items + 1)."""
        hidden = self.encode(seqs)
        return self.output_proj(hidden)

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        """Generate top-k next item predictions by appending [MASK] at sequence end."""
        self.eval()
        device = next(self.parameters()).device
        valid_history = [iid for iid in history if 0 < iid <= self.num_items]
        if not valid_history:
            return list(range(1, min(k + 1, self.num_items + 1)))

        # Append mask token to sequence end
        sub_seq = valid_history[-(self.max_seq_len - 1) :] + [self.mask_token]
        pad_len = self.max_seq_len - len(sub_seq)
        padded_seq = sub_seq + [0] * pad_len
        mask_pos = len(sub_seq) - 1

        seq_tensor = torch.tensor([padded_seq], dtype=torch.long, device=device)

        with torch.no_grad():
            hidden = self.encode(seq_tensor)  # (1, L, H)
            mask_hidden = hidden[0, mask_pos, :]  # (H,)
            logits = self.output_proj(mask_hidden)  # (num_items + 1)

            logits[0] = -1e9
            for iid in valid_history:
                if iid <= self.num_items:
                    logits[iid] = -1e9

            topk_indices = torch.topk(logits, k=min(k, self.num_items)).indices.tolist()
            return [idx for idx in topk_indices if idx > 0]

    def fit(self, train_data: Any) -> None:
        """Fit model parameters on training interactions / sequences."""
        if isinstance(train_data, dict):
            train_bert4rec(train_data, model=self)


def train_bert4rec(
    sequences: dict[int, list[int]],
    model: BERT4Rec | None = None,
    num_items: int | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_dim: int = 64,
    seed: int = SEED,
) -> BERT4Rec:
    """Train BERT4Rec model using Cloze masked item prediction."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if num_items is None:
        num_items = max(max(seq) for seq in sequences.values() if seq)

    if model is None:
        model = BERT4Rec(
            num_items=num_items,
            hidden_dim=hidden_dim,
            num_heads=2,
            num_layers=2,
        )

    dataset = BERT4RecDataset(sequences, num_items=num_items, max_seq_len=model.max_seq_len)
    if len(dataset) == 0:
        return model

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(epochs):
        for seqs, labels in dataloader:
            optimizer.zero_grad()
            logits = model(seqs)  # (B, L, num_items + 1)
            # Flatten to compute loss over non-zero label positions
            loss = criterion(logits.view(-1, model.num_items + 1), labels.view(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    return model
