from __future__ import annotations

import pandas as pd

from pipeline.data.splits import split_leave_two_out, split_temporal


def test_split_leave_two_out():
    sequences = {
        1: [10, 20, 30, 40, 50],
        2: [100, 200, 300],
        3: [1, 2],  # Too short (< 3)
    }

    splits = split_leave_two_out(sequences, min_length=3)

    assert 3 not in splits.train_sequences
    assert splits.train_sequences[1] == [10, 20, 30]
    assert splits.val_targets[1] == 40
    assert splits.test_targets[1] == 50
    assert splits.val_history[1] == [10, 20, 30]
    assert splits.test_history[1] == [10, 20, 30, 40]

    assert splits.train_sequences[2] == [100]
    assert splits.val_targets[2] == 200
    assert splits.test_targets[2] == 300
    assert splits.val_history[2] == [100]
    assert splits.test_history[2] == [100, 200]


def test_split_temporal():
    # Construct synthetic interactions chronologically
    records = []
    # User 1 has 5 train items, 1 val item, 1 test item
    for i in range(5):
        records.append({"user_id": 1, "item_id": 10 + i, "rating": 4.0, "timestamp": 100 + i})
    records.append({"user_id": 1, "item_id": 99, "rating": 4.0, "timestamp": 850})  # val
    records.append({"user_id": 1, "item_id": 100, "rating": 4.0, "timestamp": 950})  # test

    # User 2 has 6 train items, 1 test item
    for i in range(6):
        records.append({"user_id": 2, "item_id": 20 + i, "rating": 4.0, "timestamp": 200 + i})
    records.append({"user_id": 2, "item_id": 200, "rating": 4.0, "timestamp": 960})  # test

    df = pd.DataFrame(records)
    splits = split_temporal(df, min_rating=3.0, min_interactions=5, val_ratio=0.15, test_ratio=0.15)

    assert 1 in splits.train_sequences
    assert len(splits.train_sequences[1]) >= 5
    assert 1 in splits.test_targets
