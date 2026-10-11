"""Steam videogames data connector."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.data.connectors.base import BaseDataConnector


class SteamConnector(BaseDataConnector):
    """Connector for Steam videogames interaction datasets.

    Supports CSV / JSON with user_id, game_id (or game_title), behavior (play/purchase),
    hours, and timestamp.
    """

    def __init__(self, data_path: Path | str) -> None:
        self.data_path = Path(data_path)
        self._user_map: dict[str, int] = {}
        self._item_map: dict[str, int] = {}
        self._items_df: pd.DataFrame | None = None
        self._interactions_df: pd.DataFrame | None = None

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._items_df is not None and self._interactions_df is not None:
            return self._items_df, self._interactions_df

        if not self.data_path.exists():
            raise FileNotFoundError(f"Steam data file not found: {self.data_path}")

        if self.data_path.suffix == ".json" or self.data_path.name.endswith(".jsonl"):
            raw_df = pd.read_json(self.data_path, lines=True)
        else:
            raw_df = pd.read_csv(self.data_path)

        # Standardize column names
        user_col = "user_id" if "user_id" in raw_df.columns else raw_df.columns[0]
        item_col = (
            "game_id"
            if "game_id" in raw_df.columns
            else ("item_id" if "item_id" in raw_df.columns else "game_title")
        )
        title_col = "game_title" if "game_title" in raw_df.columns else "title"

        # Map to integer IDs if needed
        is_int_user = pd.api.types.is_integer_dtype(raw_df[user_col])
        if not is_int_user:
            users = raw_df[user_col].unique().tolist()
            self._user_map = {u: idx + 1 for idx, u in enumerate(users)}

        is_int_item = pd.api.types.is_integer_dtype(raw_df[item_col])
        if not is_int_item:
            items = raw_df[item_col].unique().tolist()
            self._item_map = {itm: idx + 1 for idx, itm in enumerate(items)}

        # Build items
        items_dict: dict[int, str] = {}
        for _, row in raw_df.iterrows():
            iid = int(row[item_col]) if is_int_item else self._item_map[row[item_col]]
            title = (
                str(row[title_col])
                if title_col in row and pd.notna(row[title_col])
                else f"Game {iid}"
            )
            items_dict[iid] = title

        items_list = [
            {
                "item_id": iid,
                "title": title,
                "genres": "Video Game",
                "description": f"{title} — Genres: Video Game",
                "attributes": json.dumps({"category": "Game"}),
            }
            for iid, title in items_dict.items()
        ]
        self._items_df = pd.DataFrame(items_list)

        # Build interactions
        interactions_list = []
        for _, row in raw_df.iterrows():
            uid = int(row[user_col]) if is_int_user else self._user_map[row[user_col]]
            iid = int(row[item_col]) if is_int_item else self._item_map[row[item_col]]
            ev = str(row.get("behavior", row.get("event_type", "play")))
            val = float(row.get("hours", row.get("value", 1.0)))
            ts = int(row.get("timestamp", 0))

            interactions_list.append(
                {
                    "user_id": uid,
                    "item_id": iid,
                    "event_type": ev,
                    "timestamp": ts,
                    "value": val,
                }
            )
        self._interactions_df = pd.DataFrame(interactions_list)
        return self._items_df, self._interactions_df

    def load_items(self) -> pd.DataFrame:
        items, _ = self._load_data()
        return items

    def load_interactions(self) -> pd.DataFrame:
        _, interactions = self._load_data()
        return interactions
