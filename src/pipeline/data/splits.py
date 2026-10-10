from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pipeline.config import split_cfg

logger = logging.getLogger(__name__)

SplitStrategy = Literal["leave_two_out", "temporal"]


@dataclass(frozen=True)
class DataSplits:
    """Rigorous train/validation/test partitions.

    Each sequence dict maps user_id -> list of item_ids:
    - train: historical items used for model fitting.
    - val_target: item_id to predict during validation (or sequence of validation items).
    - test_target: item_id to predict during final testing (never used during training/tuning).
    - val_history: history available up to the validation split point.
    - test_history: history available up to the test split point.
    """

    train_sequences: dict[int, list[int]]
    val_targets: dict[int, int]
    test_targets: dict[int, int]
    val_history: dict[int, list[int]]
    test_history: dict[int, list[int]]


def split_leave_two_out(
    sequences: dict[int, list[int]],
    min_length: int = 3,
) -> DataSplits:
    """Leave-two-out sequential split.

    For each user with sequence [i_1, ..., i_{n-2}, i_{n-1}, i_n]:
    - test_target: i_n
    - test_history: [i_1, ..., i_{n-1}]
    - val_target: i_{n-1}
    - val_history: [i_1, ..., i_{n-2}]
    - train_sequence: [i_1, ..., i_{n-2}]
    """
    train_seqs: dict[int, list[int]] = {}
    val_targets: dict[int, int] = {}
    test_targets: dict[int, int] = {}
    val_history: dict[int, list[int]] = {}
    test_history: dict[int, list[int]] = {}

    for uid, seq in sequences.items():
        if len(seq) < min_length:
            continue
        test_item = seq[-1]
        val_item = seq[-2]
        train_items = seq[:-2]

        train_seqs[uid] = train_items
        val_targets[uid] = val_item
        test_targets[uid] = test_item
        val_history[uid] = train_items
        test_history[uid] = seq[:-1]

    logger.info(
        "Split leave-two-out across %d users (min_length=%d)",
        len(train_seqs),
        min_length,
    )
    return DataSplits(
        train_sequences=train_seqs,
        val_targets=val_targets,
        test_targets=test_targets,
        val_history=val_history,
        test_history=test_history,
    )


def split_temporal(
    ratings: pd.DataFrame,
    min_rating: float = split_cfg.min_rating,
    min_interactions: int = split_cfg.min_interactions,
    val_ratio: float = split_cfg.temporal_val_ratio,
    test_ratio: float = split_cfg.temporal_test_ratio,
) -> DataSplits:
    """Global temporal cutoff split.

    Interactions are sorted chronologically by timestamp.
    Train: [0, t_val)
    Validation: [t_val, t_test)
    Test: [t_test, end)
    """
    positive = ratings.loc[ratings["rating"] >= min_rating].copy()
    positive = positive.sort_values("timestamp", kind="stable")

    total = len(positive)
    val_idx = int(total * (1.0 - val_ratio - test_ratio))
    test_idx = int(total * (1.0 - test_ratio))

    t_val = positive.iloc[val_idx]["timestamp"]
    t_test = positive.iloc[test_idx]["timestamp"]

    train_df = positive.loc[positive["timestamp"] < t_val]
    val_df = positive.loc[(positive["timestamp"] >= t_val) & (positive["timestamp"] < t_test)]
    test_df = positive.loc[positive["timestamp"] >= t_test]

    train_grouped = (
        train_df.sort_values("timestamp", kind="stable").groupby("user_id")["item_id"].agg(list)
    )
    val_grouped = (
        val_df.sort_values("timestamp", kind="stable").groupby("user_id")["item_id"].agg(list)
    )
    test_grouped = (
        test_df.sort_values("timestamp", kind="stable").groupby("user_id")["item_id"].agg(list)
    )

    train_seqs: dict[int, list[int]] = {}
    val_targets: dict[int, int] = {}
    test_targets: dict[int, int] = {}
    val_history: dict[int, list[int]] = {}
    test_history: dict[int, list[int]] = {}

    for uid, train_items in train_grouped.items():
        uid_int = int(uid)
        items = [int(i) for i in train_items]
        if len(items) < min_interactions:
            continue
        train_seqs[uid_int] = items
        val_history[uid_int] = list(items)

        # Check if user has validation interactions
        if uid in val_grouped and len(val_grouped[uid]) > 0:
            val_items = [int(i) for i in val_grouped[uid]]
            val_targets[uid_int] = val_items[0]
            val_hist = list(items) + val_items
        else:
            val_hist = list(items)

        # Check if user has test interactions
        if uid in test_grouped and len(test_grouped[uid]) > 0:
            test_items = [int(i) for i in test_grouped[uid]]
            test_targets[uid_int] = test_items[0]
            test_history[uid_int] = val_hist
        elif uid_int in val_targets:
            test_history[uid_int] = val_hist

    logger.info(
        "Temporal split: %d train users, %d val targets, %d test targets",
        len(train_seqs),
        len(val_targets),
        len(test_targets),
    )

    return DataSplits(
        train_sequences=train_seqs,
        val_targets=val_targets,
        test_targets=test_targets,
        val_history=val_history,
        test_history=test_history,
    )
