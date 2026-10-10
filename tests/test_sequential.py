"""Tests for the SASRec model and its data pipeline (tiny synthetic data, CPU)."""

from __future__ import annotations

import torch

from pipeline.models.sequential import SASRec, SequentialDataset, pad_right, predict_next_items

MAXLEN = 8


def _model(num_items: int = 20) -> SASRec:
    torch.manual_seed(0)
    model = SASRec(num_items=num_items, hidden_dim=16, maxlen=MAXLEN, num_heads=2, num_layers=2)
    model.eval()
    return model


def test_pad_right_pads_and_truncates():
    assert pad_right([1, 2], 4) == [1, 2, 0, 0]
    assert pad_right([1, 2, 3, 4, 5], 3) == [3, 4, 5]


def test_dataset_holds_out_last_item_and_shifts_targets():
    ds = SequentialDataset({1: [5, 6, 7, 8]}, maxlen=MAXLEN)
    inp, tgt = ds[0]
    assert inp.tolist() == [5, 6, 0, 0, 0, 0, 0, 0]
    assert tgt.tolist() == [6, 7, 0, 0, 0, 0, 0, 0]


def test_short_sequence_logits_are_finite():
    model = _model()
    seq = torch.tensor([pad_right([3, 4], MAXLEN)])
    assert torch.isfinite(model.last_logits(seq)).all()


def test_prediction_depends_on_most_recent_item():
    model = _model()
    a = model.last_logits(torch.tensor([pad_right([3, 4, 5], MAXLEN)]))
    b = model.last_logits(torch.tensor([pad_right([3, 4, 9], MAXLEN)]))
    assert not torch.allclose(a, b)


def test_padding_does_not_change_prediction():
    model = _model()
    short = model.last_logits(torch.tensor([pad_right([3, 4, 5], MAXLEN)]))
    exact = SASRec(num_items=20, hidden_dim=16, maxlen=3, num_heads=2, num_layers=2)
    # Same weights; positional table differs only in size, first rows match
    state = model.state_dict()
    state["pos_emb.weight"] = state["pos_emb.weight"][:3]
    exact.load_state_dict(state)
    exact.eval()
    unpadded = exact.last_logits(torch.tensor([[3, 4, 5]]))
    assert torch.allclose(short, unpadded, atol=1e-5)


def test_predict_excludes_history_and_padding():
    model = _model()
    history = [3, 4, 5]
    recs = predict_next_items(model, history, top_k=10, exclude=set(history))
    assert len(recs) == 10
    assert 0 not in recs
    assert not set(recs) & set(history)


def test_predict_ignores_unknown_item_ids():
    model = _model(num_items=20)
    assert len(predict_next_items(model, [3, 999], top_k=5)) == 5
    assert predict_next_items(model, [999], top_k=5) == []
