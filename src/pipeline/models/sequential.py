from __future__ import annotations

import logging
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pipeline.config import (
    SEED,
    SEQ_MODEL_PATH,
    USER_SEQUENCES_PKL,
    ensure_dirs,
    eval_cfg,
    sasrec_cfg,
)
from pipeline.evaluation.metrics import hit_rate_at_k, ndcg_at_k

logger = logging.getLogger(__name__)


class SequentialDataset(Dataset):
    """
    Train mode: target = seq[-2], input = seq[:-2]  — holds out last item for test.
    Test mode:  target = seq[-1], input = seq[:-1].
    """

    def __init__(
        self,
        sequences: dict[int, list[int]],
        num_items: int,
        maxlen: int = sasrec_cfg.max_seq_len,
        mode: str = "train",
    ) -> None:
        self.maxlen = maxlen
        self.samples: list[tuple[list[int], int]] = []

        for seq in sequences.values():
            if len(seq) < 2:
                continue
            if mode == "train":
                inp, tgt = seq[:-2], seq[-2]
            else:
                inp, tgt = seq[:-1], seq[-1]

            if len(inp) > maxlen:
                inp = inp[-maxlen:]
            self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        # Left-pad with zeros so the causal mask attends to the most recent items
        padded = [0] * (self.maxlen - len(inp)) + inp
        return torch.tensor(padded, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


class SASRec(nn.Module):
    """Kang & McAuley, 2018 — self-attentive sequential recommendation."""

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = sasrec_cfg.hidden_dim,
        maxlen: int = sasrec_cfg.max_seq_len,
        num_heads: int = sasrec_cfg.num_heads,
        num_layers: int = sasrec_cfg.num_layers,
        dropout: float = sasrec_cfg.dropout,
    ) -> None:
        super().__init__()
        self.num_items = num_items

        # padding_idx=0 so zero-padded positions produce zero gradients
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _batch, seq_len = seq.shape
        device = seq.device

        item_e = self.item_emb(seq)
        pos_e = self.pos_emb(torch.arange(seq_len, device=device).unsqueeze(0))
        x = self.emb_dropout(item_e + pos_e)

        # Causal mask prevents attending to future items (autoregressive)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = seq == 0

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        # Use the last non-padding position as the sequence representation
        lengths = (seq != 0).sum(dim=1).clamp(min=1) - 1
        last_hidden = x[torch.arange(_batch, device=device), lengths]
        return self.output_proj(last_hidden)


def train_sasrec(sequences: dict[int, list[int]]) -> SASRec:
    ensure_dirs()
    all_items = {item for seq in sequences.values() for item in seq}
    num_items = max(all_items)

    logger.info(
        "Training SASRec — %d items, %d users, %d epochs",
        num_items,
        len(sequences),
        sasrec_cfg.epochs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_ds = SequentialDataset(sequences, num_items, mode="train")
    loader = DataLoader(train_ds, batch_size=sasrec_cfg.batch_size, shuffle=True, num_workers=0)

    model = SASRec(num_items=num_items).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=sasrec_cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(1, sasrec_cfg.epochs + 1):
        total_loss = 0.0
        for batch_seq, batch_tgt in loader:
            logits = model(batch_seq.to(device))
            loss = criterion(logits, batch_tgt.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("  Epoch %3d/%d  loss=%.4f", epoch, sasrec_cfg.epochs, avg)

    torch.save({"model_state": model.state_dict(), "num_items": num_items}, str(SEQ_MODEL_PATH))
    logger.info("SASRec checkpoint → %s", SEQ_MODEL_PATH)
    return model


def load_sasrec_model() -> tuple[SASRec, int]:
    ckpt = torch.load(str(SEQ_MODEL_PATH), map_location="cpu", weights_only=False)
    model = SASRec(num_items=ckpt["num_items"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["num_items"]


def predict_next_items(
    model: SASRec,
    seq: list[int],
    top_k: int = 10,
    exclude: set[int] | None = None,
) -> list[int]:
    device = next(model.parameters()).device
    maxlen = sasrec_cfg.max_seq_len

    if len(seq) > maxlen:
        seq = seq[-maxlen:]
    padded = [0] * (maxlen - len(seq)) + seq
    inp = torch.tensor([padded], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(inp)

    # Mask out the padding token (idx 0) so it's never recommended
    logits[0, 0] = -float("inf")
    if exclude:
        for idx in exclude:
            if 0 < idx < logits.shape[1]:
                logits[0, idx] = -float("inf")

    return logits[0].topk(top_k).indices.cpu().tolist()


def evaluate_sasrec(sequences: dict[int, list[int]], k: int = 10) -> dict[str, float]:
    """Leave-last-one-out: train on seq[:-1], test on seq[-1]."""
    model, _ = load_sasrec_model()
    hits, ndcgs = [], []

    for seq in tqdm(sequences.values(), desc=f"Eval SASRec @{k}"):
        if len(seq) < 2:
            continue
        recs = predict_next_items(model, seq[:-1], top_k=k, exclude=set(seq[:-1]))
        hits.append(hit_rate_at_k(recs, seq[-1], k))
        ndcgs.append(ndcg_at_k(recs, seq[-1], k))

    result = {
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
    logger.info(
        "SASRec — HR@%d: %.4f  NDCG@%d: %.4f", k, result[f"HitRate@{k}"], k, result[f"NDCG@{k}"]
    )
    return result


def run_phase3(sequences: dict[int, list[int]] | None = None) -> dict[str, float]:
    if sequences is None:
        with open(USER_SEQUENCES_PKL, "rb") as f:
            sequences = pickle.load(f)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_sasrec(sequences)

    results: dict[str, float] = {}
    for k in eval_cfg.k_values:
        results.update(evaluate_sasrec(sequences, k=k))
    return results
