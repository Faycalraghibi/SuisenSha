from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Recommender(ABC):
    """Abstract base protocol for recommendation models."""

    @abstractmethod
    def fit(self, train_data: Any) -> None:
        """Fit model parameters on training interactions / sequences."""
        raise NotImplementedError

    @abstractmethod
    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        """Generate top-k recommendations given user interaction history."""
        raise NotImplementedError
