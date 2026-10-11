"""Base connector interface for the data platform."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from pipeline.data.contracts import (
    validate_interactions,
    validate_items,
)

logger = logging.getLogger(__name__)


class BaseDataConnector(ABC):
    """Abstract base class for all data connectors in SuisenSha."""

    @abstractmethod
    def load_items(self) -> pd.DataFrame:
        """Load items conforming to the items contract.

        Returns:
            DataFrame with columns: item_id, title, [genres, description, attributes]
        """
        ...

    @abstractmethod
    def load_interactions(self) -> pd.DataFrame:
        """Load user-item interactions conforming to the interactions contract.

        Returns:
            DataFrame with columns: user_id, item_id, event_type, timestamp, [value]
        """
        ...

    def validate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate both items and interactions DataFrames.

        Returns:
            Tuple of (validated_items, validated_interactions)
        """
        items = validate_items(self.load_items())
        interactions = validate_interactions(self.load_interactions())
        return items, interactions

    def build_sequences(
        self,
        min_interactions: int = 5,
        min_rating: float | None = None,
        positive_events: list[str] | None = None,
    ) -> dict[int, list[int]]:
        """Construct chronologically ordered user item sequences.

        Args:
            min_interactions: Minimum number of positive events for a user to be included.
            min_rating: If set, filters interactions where `value >= min_rating`.
            positive_events: If set, filters interactions where `event_type` is in this list.

        Returns:
            Dictionary mapping user_id -> list of item_ids ordered by timestamp.
        """
        _, interactions = self.validate()
        df = interactions.copy()

        if min_rating is not None and "value" in df.columns:
            df = df.loc[df["value"] >= min_rating]

        if positive_events is not None:
            df = df.loc[df["event_type"].isin(positive_events)]

        # Stable sort preserving original ordering for timestamp ties
        df = df.sort_values(["user_id", "timestamp"], kind="stable")
        grouped = df.groupby("user_id", sort=True)["item_id"].agg(list)

        sequences = {
            int(u): [int(i) for i in seq]
            for u, seq in grouped.items()
            if len(seq) >= min_interactions
        }
        return sequences

    def export_csv(
        self, items_path: Path | str, interactions_path: Path | str
    ) -> tuple[Path, Path]:
        """Validate and export items and interactions to CSV files."""
        items, interactions = self.validate()
        p_items = Path(items_path)
        p_inter = Path(interactions_path)
        p_items.parent.mkdir(parents=True, exist_ok=True)
        p_inter.parent.mkdir(parents=True, exist_ok=True)

        items.to_csv(p_items, index=False)
        interactions.to_csv(p_inter, index=False)
        logger.info("Exported %d items to %s", len(items), p_items)
        logger.info("Exported %d interactions to %s", len(interactions), p_inter)
        return p_items, p_inter

    def export_parquet(
        self, items_path: Path | str, interactions_path: Path | str
    ) -> tuple[Path, Path]:
        """Validate and export items and interactions to Parquet files."""
        items, interactions = self.validate()
        p_items = Path(items_path)
        p_inter = Path(interactions_path)
        p_items.parent.mkdir(parents=True, exist_ok=True)
        p_inter.parent.mkdir(parents=True, exist_ok=True)

        items.to_parquet(p_items, index=False)
        interactions.to_parquet(p_inter, index=False)
        logger.info("Exported %d items to %s (parquet)", len(items), p_items)
        logger.info("Exported %d interactions to %s (parquet)", len(interactions), p_inter)
        return p_items, p_inter
