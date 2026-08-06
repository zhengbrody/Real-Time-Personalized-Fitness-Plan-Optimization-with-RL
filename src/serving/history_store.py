"""
Training-History Store
======================

Supplies the daily history a cache miss needs in order to recompute a user's
rolling-window features.

There are two implementations because they answer two different needs:

:class:`ParquetHistoryStore`
    Reads the table the PySpark pipeline writes.  This is the production path.

:class:`SyntheticHistoryStore`
    Generates deterministic history on demand.  Used by the latency benchmark
    and local demos, where the point is to exercise the *cost* of the recompute
    path rather than any particular user's data.  It is a separate class rather
    than a fallback inside the real store on purpose: a production store that
    quietly invents data when a user is missing is a much worse failure than
    one that returns empty.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "HistoryStore",
    "ParquetHistoryStore",
    "SyntheticHistoryStore",
    "EMPTY_HISTORY",
]

HISTORY_COLUMNS = ("hrv", "resting_hr", "sleep_hours", "load", "completed")

EMPTY_HISTORY = pd.DataFrame(columns=list(HISTORY_COLUMNS))


class HistoryStore(Protocol):
    """Anything that can return a user's daily history, oldest row first."""

    def get(self, user_id: str) -> pd.DataFrame:  # pragma: no cover - protocol
        ...


class ParquetHistoryStore:
    """
    Reads history from the parquet table produced by the offline pipeline.

    Loads once and indexes by user, because a per-request parquet scan would
    cost far more than the recompute the cache is meant to avoid.
    """

    def __init__(self, path: Path, days: int = 90):
        self.path = Path(path)
        self.days = int(days)
        self._by_user: dict[str, pd.DataFrame] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            logger.warning(
                "history table %s not found; serving empty history", self.path
            )
            return
        try:
            table = pd.read_parquet(self.path)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to read history table %s (%s)", self.path, exc)
            return

        if "user_id" not in table.columns:
            logger.error("history table %s has no user_id column", self.path)
            return

        sort_key = "date" if "date" in table.columns else None
        for user_id, group in table.groupby("user_id"):
            if sort_key:
                group = group.sort_values(sort_key)
            self._by_user[str(user_id)] = group.tail(self.days).reset_index(drop=True)
        logger.info(
            "loaded history for %d users from %s", len(self._by_user), self.path
        )

    def get(self, user_id: str) -> pd.DataFrame:
        return self._by_user.get(str(user_id), EMPTY_HISTORY)


class SyntheticHistoryStore:
    """
    Deterministic synthetic history, derived from a hash of the user id.

    For benchmarking and demos only.  Every user id yields a stable history, so
    a cold-cache measurement is repeatable, and distinct ids yield distinct
    histories, so a benchmark can force a cache miss on every request simply by
    using a fresh id.
    """

    def __init__(self, days: int = 90):
        self.days = int(days)

    def get(self, user_id: str) -> pd.DataFrame:
        seed = abs(hash(str(user_id))) % (2**31)
        rng = np.random.default_rng(seed)
        n = self.days
        return pd.DataFrame(
            {
                "hrv": rng.normal(52.0, 7.0, n),
                "resting_hr": rng.normal(58.0, 4.0, n),
                "sleep_hours": np.clip(rng.normal(7.2, 0.8, n), 4.0, 10.0),
                "load": rng.choice([0.0, 0.0, 90.0, 180.0, 270.0], n),
                "completed": rng.integers(0, 2, n),
            }
        )
