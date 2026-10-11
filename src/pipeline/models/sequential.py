from __future__ import annotations

import copy
import logging
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.config import (
    SEED,
    SEQ_MODEL_PATH,
    ensure_dirs,
    eval_cfg,
    sasrec_cfg,
)
from pipeline.data.loader import load_sequences
from pipeline.data.splits import split_leave_two_out
from pipeline.evaluation.metrics import evaluate_leave_one_out, ndcg_at_k
from pipeline.models.base import Recommender

logger = logging.getLogger(__name__)


def pad_right(seq: list[int], maxlen: int) -> list[int]:
    """Keep the most recent ``maxlen`` items and right-pad with zeros.

    Right padding means every real position only attends (causally) to earlier
    real positions, so no attention row is ever fully masked.
    """
    seq = seq[-maxlen:]
    return seq + [0] * (maxlen - len(seq))


def generate_sliding_windows(
    seq: list[int],
    maxlen: int = sasrec_cfg.max_seq_len,
    stride: int = sasrec_cfg.window_stride,
) -> list[list[int]]:
    """Slice sequences longer than maxlen into overlapping sub-sequences."""
    if len(seq) <= maxlen:
        return [seq]
    windows = []
    for start in range(0, len(seq) - 1, stride):
        sub = seq[start : start + maxlen + 1]
        if len(sub) >= 2:
            windows.append(sub)
    if not windows or windows[-1] != seq[-maxlen - 1 :]:
        windows.append(seq[-maxlen - 1 :])
    return windows


class SequentialDataset(Dataset):
    """Next-item training pairs over sequences with sliding window augmentation."""

    def __init__(
        self,
        sequences: dict[int, list[int]],
        maxlen: int = sasrec_cfg.max_seq_len,
        stride: int = sasrec_cfg.window_stride,
        hold_out_last: bool = True,
    ) -> None:
        self.maxlen = maxlen
        self.samples: list[tuple[list[int], list[int]]] = []

        for seq in sequences.values():
            work_seq = seq[:-1] if hold_out_last else seq
            if len(work_seq) < 2:
                continue
            windows = generate_sliding_windows(work_seq, maxlen=maxlen, stride=stride)
            for window in windows:
                if len(window) < 2:
                    continue
                self.samples.append((window[:-1], window[1:]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        return (
            torch.tensor(pad_right(inp, self.maxlen), dtype=torch.long),
            torch.tensor(pad_right(tgt, self.maxlen), dtype=torch.long),
        )


class SASRec(nn.Module, Recommender):
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
        self.maxlen = maxlen

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
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_items + 1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Hidden state at every position: (batch, seq_len, hidden_dim)."""
        seq_len = seq.shape[1]
        device = seq.device

        item_e = self.item_emb(seq)
        pos_e = self.pos_emb(torch.arange(seq_len, device=device).unsqueeze(0))
        x = self.emb_dropout(item_e + pos_e)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.layer_norm(x)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Logits for the next item at every position: (batch, seq_len, num_items + 1)."""
        return self.output_proj(self.encode(seq))

    def last_logits(self, seq: torch.Tensor) -> torch.Tensor:
        """Next-item logits after the last real (non-padding) item: (batch, num_items + 1)."""
        hidden = self.encode(seq)
        last = (seq != 0).sum(dim=1).clamp(min=1) - 1
        return self.output_proj(hidden[torch.arange(seq.shape[0], device=seq.device), last])

    def fit(self, train_data: Any) -> None:
        """Recommender protocol fit implementation."""
        if isinstance(train_data, dict):
            train_sasrec(train_data)

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        """Recommender protocol recommend implementation."""
        return predict_next_items(self, history, top_k=k, exclude=set(history))


def compute_val_ndcg(
    model: SASRec,
    val_history: dict[int, list[int]],
    val_targets: dict[int, int],
    top_k: int = 10,
) -> float:
    """Evaluate mean NDCG@k on validation split."""
    scores = []
    for uid, target in val_targets.items():
        hist = val_history.get(uid, [])
        recs = predict_next_items(model, hist, top_k=top_k, exclude=set(hist))
        scores.append(ndcg_at_k(recs, target, k=top_k))
    return float(np.mean(scores)) if scores else 0.0


def train_sasrec(
    sequences: dict[int, list[int]],
    val_history: dict[int, list[int]] | None = None,
    val_targets: dict[int, int] | None = None,
    seed: int = SEED,
) -> SASRec:
    ensure_dirs()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If validation split is not explicitly passed, partition train and val
    if val_history is None or val_targets is None:
        splits = split_leave_two_out(sequences)
        train_seqs = splits.train_sequences
        v_hist = splits.val_history
        v_tgts = splits.val_targets
    else:
        train_seqs = sequences
        v_hist = val_history
        v_tgts = val_targets

    all_items = set()
    for seq in train_seqs.values():
        all_items.update(seq)
    for seq in v_hist.values():
        all_items.update(seq)
    all_items.update(v_tgts.values())

    num_items = max(all_items) if all_items else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Training SASRec — %d items, %d train sequences, patience=%d, device=%s",
        num_items,
        len(train_seqs),
        sasrec_cfg.patience,
        device,
    )

    train_ds = SequentialDataset(
        train_seqs,
        maxlen=sasrec_cfg.max_seq_len,
        stride=sasrec_cfg.window_stride,
        hold_out_last=False,
    )
    loader = DataLoader(train_ds, batch_size=sasrec_cfg.batch_size, shuffle=True, num_workers=0)

    model = SASRec(num_items=num_items).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=sasrec_cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_ndcg = -1.0
    best_weights: dict[str, Any] | None = None
    patience_counter = 0

    for epoch in range(1, sasrec_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_seq, batch_tgt in loader:
            logits = model(batch_seq.to(device))
            loss = criterion(logits.reshape(-1, num_items + 1), batch_tgt.to(device).reshape(-1))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)

        # Validation check for early stopping
        model.eval()
        val_ndcg = compute_val_ndcg(model, v_hist, v_tgts, top_k=10)

        if epoch % 5 == 0 or epoch == 1 or val_ndcg > best_val_ndcg:
            logger.info(
                "  Epoch %3d/%d  loss=%.4f  val_ndcg@10=%.4f",
                epoch,
                sasrec_cfg.epochs,
                avg_loss,
                val_ndcg,
            )

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= sasrec_cfg.patience:
                logger.info(
                    "Early stopping triggered at epoch %d (best val_ndcg@10=%.4f)",
                    epoch,
                    best_val_ndcg,
                )
                break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    torch.save({"model_state": model.state_dict(), "num_items": num_items}, str(SEQ_MODEL_PATH))
    logger.info("Best SASRec checkpoint saved → %s", SEQ_MODEL_PATH)
    return model


def load_sasrec_model() -> tuple[SASRec, int]:
    ckpt = torch.load(str(SEQ_MODEL_PATH), map_location="cpu", weights_only=True)
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
    seq = [i for i in seq if 0 < i <= model.num_items]
    if not seq:
        return []
    inp = torch.tensor([pad_right(seq, model.maxlen)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model.last_logits(inp)[0]

    logits[0] = -float("inf")
    if exclude:
        idx = [i for i in exclude if 0 < i < logits.shape[0]]
        logits[idx] = -float("inf")

    return logits.topk(min(top_k, model.num_items)).indices.cpu().tolist()


def evaluate_sasrec(
    sequences: dict[int, list[int]],
    k_values: tuple[int, ...] = eval_cfg.k_values,
    model: SASRec | None = None,
) -> dict[str, float]:
    if model is None:
        model, _ = load_sasrec_model()
    model.eval()

    def recommend(history: list[int], top_k: int) -> list[int]:
        return predict_next_items(model, history, top_k=top_k, exclude=set(history))

    result = evaluate_leave_one_out(sequences, recommend, k_values, desc="Eval SASRec")
    logger.info("SASRec — %s", result)
    return result


def run_phase3(sequences: dict[int, list[int]] | None = None) -> dict[str, float]:
    if sequences is None:
        sequences = load_sequences()

    model = train_sasrec(sequences)
    return evaluate_sasrec(sequences, model=model)
