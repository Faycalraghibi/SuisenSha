"""SQL-based data connector supporting PostgreSQL, SQLite, and other relational databases."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from pipeline.data.connectors.base import BaseDataConnector


class SQLConnector(BaseDataConnector):
    """Data connector for reading and writing tables via SQLAlchemy."""

    def __init__(
        self,
        connection_uri: str,
        items_table: str = "items",
        interactions_table: str = "interactions",
    ) -> None:
        self.connection_uri = connection_uri
        self.items_table = items_table
        self.interactions_table = interactions_table
        self._engine = create_engine(connection_uri)

    def load_items(self) -> pd.DataFrame:
        with self._engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {self.items_table}", conn)
        self._engine.dispose()
        return df

    def load_interactions(self) -> pd.DataFrame:
        with self._engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {self.interactions_table}", conn)
        self._engine.dispose()
        return df

    def save_tables(self, items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        """Write items and interactions DataFrames into the database tables."""
        with self._engine.begin() as conn:
            items_df.to_sql(self.items_table, conn, if_exists="replace", index=False)
            interactions_df.to_sql(self.interactions_table, conn, if_exists="replace", index=False)
        self._engine.dispose()
