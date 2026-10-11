"""Amazon Reviews (Books) data connector."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.data.connectors.base import BaseDataConnector


class AmazonReviewsConnector(BaseDataConnector):
    """Connector for Amazon Reviews dataset format (e.g. Books category).

    Accepts JSON lines or CSV files with columns:
    reviewerID, asin, overall, unixReviewTime, [reviewText, title]
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
            raise FileNotFoundError(f"Amazon data file not found: {self.data_path}")

        records: list[dict] = []
        if self.data_path.suffix == ".json" or self.data_path.name.endswith(".jsonl"):
            with open(self.data_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            raw_df = pd.DataFrame(records)
        else:
            raw_df = pd.read_csv(self.data_path)

        # Map string IDs to continuous integers
        users = raw_df["reviewerID"].unique().tolist()
        self._user_map = {uid: idx + 1 for idx, uid in enumerate(users)}

        items = raw_df["asin"].unique().tolist()
        self._item_map = {asin: idx + 1 for idx, asin in enumerate(items)}

        # Build items DataFrame
        titles: dict[int, str] = {}
        for _, row in raw_df.iterrows():
            iid = self._item_map[row["asin"]]
            title = str(row.get("title", f"Book {row['asin']}"))
            if iid not in titles or titles[iid].startswith("Book "):
                titles[iid] = title

        items_list = []
        for asin, iid in self._item_map.items():
            t = titles.get(iid, f"Book {asin}")
            items_list.append(
                {
                    "item_id": iid,
                    "title": t,
                    "genres": "Books",
                    "description": f"{t} — Genres: Books",
                    "attributes": json.dumps({"asin": asin}),
                }
            )
        self._items_df = pd.DataFrame(items_list)

        # Build interactions DataFrame
        interactions_list = []
        for _, row in raw_df.iterrows():
            uid = self._user_map[row["reviewerID"]]
            iid = self._item_map[row["asin"]]
            ts = int(row.get("unixReviewTime", 0))
            rating = float(row.get("overall", 5.0))
            interactions_list.append(
                {
                    "user_id": uid,
                    "item_id": iid,
                    "event_type": "rating",
                    "timestamp": ts,
                    "value": rating,
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
