"""Data connectors for SuisenSha data platform."""

from __future__ import annotations

from pipeline.data.connectors.amazon import AmazonReviewsConnector
from pipeline.data.connectors.base import BaseDataConnector
from pipeline.data.connectors.file import CSVConnector, ParquetConnector
from pipeline.data.connectors.movielens import MovieLensConnector
from pipeline.data.connectors.sql import SQLConnector
from pipeline.data.connectors.steam import SteamConnector

__all__ = [
    "AmazonReviewsConnector",
    "BaseDataConnector",
    "CSVConnector",
    "MovieLensConnector",
    "ParquetConnector",
    "SQLConnector",
    "SteamConnector",
]
