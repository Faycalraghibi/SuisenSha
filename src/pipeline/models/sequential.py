from __future__ import annotations

import logging
import random

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
from pipeline.evaluation.metrics import evaluate_leave_one_out

logger = logging.getLogger(__name__)


def pad_right(seq: list[int], maxlen: int) -> list[int]:
    """Keep the most recent ``maxlen`` items and right-pad with zeros.

    Right padding means every real position only attends (causally) to earlier
    real positions, so no attention row is ever fully masked.
    """
    seq = seq[-maxlen:]
    return seq + [0] * (maxlen - len(seq))


class SequentialDataset(Dataset):
    """Next-item training pairs over every position of each user's sequence.

    The last item of each sequence is held out for evaluation, so training
    uses ``seq[:-1]``: inputs ``seq[:-2]`` and targets shifted by one.
    """

    def __init__(
        self,
        sequences: dict[int, list[int]],
        maxlen: int = sasrec_cfg.max_seq_len,
    ) -> None:
        self.maxlen = maxlen
        self.samples: list[tuple[list[int], list[int]]] = []

        for seq in sequences.values():
            train_seq = seq[:-1]
            if len(train_seq) < 2:
                continue
            self.samples.append((train_seq[:-1], train_seq[1:]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        return (
            torch.tensor(pad_right(inp, self.maxlen), dtype=torch.long),
            # Padded targets are 0, which the loss ignores
            torch.tensor(pad_right(tgt, self.maxlen), dtype=torch.long),
        )


class SASRec(nn.Module):
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

        # Causal mask prevents attending to future items (autoregressive).
        # With right padding it also keeps real positions from seeing padding.
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


def train_sasrec(sequences: dict[int, list[int]]) -> SASRec:
    ensure_dirs()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    num_items = max(item for seq in sequences.values() for item in seq)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Training SASRec — %d items, %d users, %d epochs, device=%s",
        num_items,
        len(sequences),
        sasrec_cfg.epochs,
        device,
    )

    train_ds = SequentialDataset(sequences)
    loader = DataLoader(train_ds, batch_size=sasrec_cfg.batch_size, shuffle=True, num_workers=0)

    model = SASRec(num_items=num_items).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=sasrec_cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(1, sasrec_cfg.epochs + 1):
        total_loss = 0.0
        for batch_seq, batch_tgt in loader:
            logits = model(batch_seq.to(device))
            loss = criterion(logits.reshape(-1, num_items + 1), batch_tgt.to(device).reshape(-1))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("  Epoch %3d/%d  loss=%.4f", epoch, sasrec_cfg.epochs, avg)

    model.eval()
    torch.save({"model_state": model.state_dict(), "num_items": num_items}, str(SEQ_MODEL_PATH))
    logger.info("SASRec checkpoint → %s", SEQ_MODEL_PATH)
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
    # Unknown ids would index past the embedding table
    seq = [i for i in seq if 0 < i <= model.num_items]
    if not seq:
        return []
    inp = torch.tensor([pad_right(seq, model.maxlen)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model.last_logits(inp)[0]

    # Mask out the padding token (idx 0) so it's never recommended
    logits[0] = -float("inf")
    if exclude:
        idx = [i for i in exclude if 0 < i < logits.shape[0]]
        logits[idx] = -float("inf")

    return logits.topk(top_k).indices.cpu().tolist()


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
