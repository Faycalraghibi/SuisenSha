"""File-based data connectors for CSV and Parquet formats."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.data.connectors.base import BaseDataConnector


class CSVConnector(BaseDataConnector):
    """Data connector for reading and writing CSV files conforming to contracts."""

    def __init__(self, items_path: Path | str, interactions_path: Path | str) -> None:
        self.items_path = Path(items_path)
        self.interactions_path = Path(interactions_path)

    def load_items(self) -> pd.DataFrame:
        if not self.items_path.exists():
            raise FileNotFoundError(f"Items CSV not found: {self.items_path}")
        df = pd.read_csv(self.items_path)
        return df

    def load_interactions(self) -> pd.DataFrame:
        if not self.interactions_path.exists():
            raise FileNotFoundError(f"Interactions CSV not found: {self.interactions_path}")
        df = pd.read_csv(self.interactions_path)
        return df


class ParquetConnector(BaseDataConnector):
    """Data connector for reading and writing Parquet files conforming to contracts."""

    def __init__(self, items_path: Path | str, interactions_path: Path | str) -> None:
        self.items_path = Path(items_path)
        self.interactions_path = Path(interactions_path)

    def load_items(self) -> pd.DataFrame:
        if not self.items_path.exists():
            raise FileNotFoundError(f"Items Parquet not found: {self.items_path}")
        df = pd.read_parquet(self.items_path)
        return df

    def load_interactions(self) -> pd.DataFrame:
        if not self.interactions_path.exists():
            raise FileNotFoundError(f"Interactions Parquet not found: {self.interactions_path}")
        df = pd.read_parquet(self.interactions_path)
        return df
