from __future__ import annotations

import torch

from pipeline.models.base import Recommender
from pipeline.models.sequential import (
    SASRec,
    SequentialDataset,
    generate_sliding_windows,
    pad_right,
    train_sasrec,
)


def test_pad_right():
    seq = [1, 2, 3]
    padded = pad_right(seq, maxlen=5)
    assert padded == [1, 2, 3, 0, 0]

    long_seq = [1, 2, 3, 4, 5, 6]
    truncated = pad_right(long_seq, maxlen=4)
    assert truncated == [3, 4, 5, 6]


def test_generate_sliding_windows():
    # Sequence shorter than maxlen
    seq = [1, 2, 3, 4]
    windows = generate_sliding_windows(seq, maxlen=5, stride=2)
    assert len(windows) == 1
    assert windows[0] == [1, 2, 3, 4]

    # Sequence longer than maxlen
    long_seq = list(range(1, 20))  # 19 items
    windows = generate_sliding_windows(long_seq, maxlen=10, stride=5)
    assert len(windows) > 1
    for w in windows:
        assert len(w) <= 11  # maxlen + 1


def test_sequential_dataset_sliding_windows():
    sequences = {1: list(range(1, 15)), 2: [1, 2, 3]}
    ds = SequentialDataset(sequences, maxlen=5, stride=3)
    assert len(ds) > 2
    inp, tgt = ds[0]
    assert inp.shape == torch.Size([5])
    assert tgt.shape == torch.Size([5])


def test_sasrec_recommender_interface():
    model = SASRec(num_items=10, hidden_dim=16, maxlen=5, num_heads=2, num_layers=1)
    assert isinstance(model, Recommender)
    recs = model.recommend(history=[1, 2], k=3)
    assert len(recs) <= 3
    assert 1 not in recs
    assert 2 not in recs


def test_train_sasrec_early_stopping(tmp_path, monkeypatch):
    import dataclasses

    import pipeline.config as cfg
    import pipeline.models.sequential as seq_mod

    monkeypatch.setattr(cfg, "SEQ_MODEL_PATH", tmp_path / "sasrec_test.pt")
    monkeypatch.setattr(seq_mod, "SEQ_MODEL_PATH", tmp_path / "sasrec_test.pt")

    new_sasrec = dataclasses.replace(
        cfg.sasrec_cfg,
        epochs=3,
        patience=2,
        hidden_dim=16,
        max_seq_len=5,
    )
    monkeypatch.setattr(cfg, "sasrec_cfg", new_sasrec)
    monkeypatch.setattr(seq_mod, "sasrec_cfg", new_sasrec)

    sequences = {
        1: [1, 2, 3, 4, 5],
        2: [2, 3, 4, 5, 6],
        3: [1, 3, 5, 7, 8],
    }

    model = train_sasrec(sequences)
    assert isinstance(model, SASRec)
    assert (tmp_path / "sasrec_test.pt").is_file()
