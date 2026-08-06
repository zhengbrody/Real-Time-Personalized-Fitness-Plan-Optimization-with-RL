"""
Online Feature Service
======================

The serving-side half of the feature path: assemble a model input vector for a
request, using Redis to cache the expensive part.

What is expensive and what is not
---------------------------------
A request carries *today's* signals — this morning's HRV, last night's sleep,
the user's self-reported fatigue.  Those are free; they arrive in the payload.

The rolling-window features are not free.  ``hrv_28d_mean``, ``acwr``,
``completion_rate_28d`` and friends are aggregates over the user's training
history, and recomputing them per request means pulling ~90 days of rows and
running a set of pandas window operations on every single call.  That work
produces the same answer all day: the windows only move when new history lands,
which happens once per day per user.

So the rolling block is computed once, cached in Redis under the user's key, and
merged with the request's fresh signals on each call.  The cache holds *derived
aggregates*, never raw health data.

Correctness constraint
----------------------
Whatever this module assembles is handed to the same
:func:`src.feature_store.transform.transform` the offline benchmark uses.  The
cache changes *when* the rolling inputs are computed, never *what* the transform
does with them — which is why a cache hit and a cache miss must produce
byte-identical vectors.  ``tests/test_feature_parity.py`` asserts exactly that,
alongside offline/online parity.

Redis is optional.  With no server reachable the service degrades to computing
rolling features on every request: slower, identical output.  A recommender
that changes its answer when a cache is cold is broken, not degraded.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

from src.feature_store.transform import FEATURE_DIM, transform

logger = logging.getLogger(__name__)

__all__ = ["FeatureService", "FeatureFetch", "compute_rolling_features"]

# Fields the cache is responsible for.  Everything else comes from the request.
ROLLING_FIELDS = (
    "hrv_7d_mean",
    "hrv_28d_mean",
    "hrv_28d_std",
    "resting_hr_baseline",
    "sleep_debt_7d",
    "acwr",
    "load_7d",
    "load_28d_mean",
    "days_since_training",
    "consecutive_hard_days",
    "completion_rate_7d",
    "completion_rate_28d",
    "streak",
    "sessions_7d",
    "days_active_28d",
)


@dataclass
class FeatureFetch:
    """A feature vector plus how it was obtained."""

    features: np.ndarray
    cache_hit: bool
    rolling_source: str  # "redis" | "recomputed" | "request"
    elapsed_ms: float
    rolling: Dict[str, float] = field(default_factory=dict)


def compute_rolling_features(history: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the rolling-window block from a user's training history.

    Parameters
    ----------
    history
        One row per day, most recent last, with columns ``hrv``, ``resting_hr``,
        ``sleep_hours``, ``load`` and ``completed``.

    Returns
    -------
    dict
        The :data:`ROLLING_FIELDS` block, ready to merge into a raw state.

    This is the work the cache exists to avoid: on a 90-day history it is a
    dozen pandas window reductions, and it produces the same answer for every
    request on a given day.
    """
    if history.empty:
        return {field_: 0.0 for field_ in ROLLING_FIELDS}

    hrv = history["hrv"].astype(float)
    load = history["load"].astype(float)
    completed = history["completed"].astype(float)
    sleep_hours = history["sleep_hours"].astype(float)
    resting_hr = history["resting_hr"].astype(float)

    last7_load = load.tail(7)
    last28_load = load.tail(28)
    chronic = float(last28_load.mean()) if len(last28_load) else 0.0
    acute = float(last7_load.mean()) if len(last7_load) else 0.0

    # Days since the last session with real load.
    days_since = 0
    for value in reversed(load.tolist()):
        if value > 1.0:
            break
        days_since += 1

    # Current streak of consecutive active days.
    streak = 0
    for value in reversed(load.tolist()):
        if value > 1.0:
            streak += 1
        else:
            break

    # Consecutive recent days at meaningful load.
    consec_hard = 0
    for value in reversed(load.tolist()):
        if value >= 180.0:
            consec_hard += 1
        else:
            break

    return {
        "hrv_7d_mean": float(hrv.tail(7).mean()),
        "hrv_28d_mean": float(hrv.tail(28).mean()),
        "hrv_28d_std": float(hrv.tail(28).std(ddof=0)) if len(hrv) > 1 else 0.0,
        "resting_hr_baseline": float(resting_hr.tail(28).mean()),
        "sleep_debt_7d": float((7.5 - sleep_hours.tail(7)).sum()),
        "acwr": float(acute / chronic) if chronic > 1e-6 else 1.0,
        "load_7d": float(last7_load.sum()),
        "load_28d_mean": chronic,
        "days_since_training": float(min(days_since, 7)),
        "consecutive_hard_days": float(min(consec_hard, 5)),
        "completion_rate_7d": float(completed.tail(7).mean()),
        "completion_rate_28d": float(completed.tail(28).mean()),
        "streak": float(min(streak, 14)),
        "sessions_7d": float((last7_load > 1.0).sum()),
        "days_active_28d": float((last28_load > 1.0).sum()),
    }


class FeatureService:
    """
    Assembles model inputs for the serving path, with an optional Redis cache
    over the rolling-window block.

    Parameters
    ----------
    redis_client
        A ``redis.Redis`` instance, or ``None`` to auto-connect from
        ``REDIS_HOST``/``REDIS_PORT``.  Pass ``False`` to disable caching
        entirely (used by the uncached arm of the latency benchmark).
    ttl_seconds
        Cache lifetime.  Defaults to 6 hours: long enough that a user's repeated
        requests within a day all hit, short enough that a stale entry cannot
        survive a day boundary and describe the wrong window.
    """

    def __init__(
        self,
        redis_client: Any = None,
        ttl_seconds: int = 6 * 3600,
        namespace: str = "profit:features",
    ):
        self.ttl_seconds = int(ttl_seconds)
        self.namespace = namespace
        self._hits = 0
        self._misses = 0

        if redis_client is False:
            self.redis = None
            self.cache_enabled = False
            return

        self.cache_enabled = True
        if redis_client is not None:
            self.redis = redis_client
        else:
            self.redis = self._connect()

    # -- connection -------------------------------------------------------

    @staticmethod
    def _connect() -> Optional[Any]:
        """Connect to Redis, returning ``None`` if it is unreachable."""
        try:
            import redis  # imported lazily: the package is optional
        except ImportError:
            logger.info("redis package not installed; feature cache disabled")
            return None

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        try:
            client = redis.Redis(
                host=host,
                port=port,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
                decode_responses=True,
            )
            client.ping()
            logger.info("feature cache connected to redis://%s:%s", host, port)
            return client
        except Exception as exc:  # noqa: BLE001 - any failure means "no cache"
            logger.info("redis unavailable (%s); serving without feature cache", exc)
            return None

    # -- cache ------------------------------------------------------------

    def _key(self, user_id: str) -> str:
        return f"{self.namespace}:{user_id}"

    def get_cached_rolling(self, user_id: str) -> Optional[Dict[str, float]]:
        """Read a user's cached rolling block, or ``None`` on miss/failure."""
        if not self.redis:
            return None
        try:
            raw = self.redis.get(self._key(user_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis read failed (%s); treating as a miss", exc)
            return None
        if raw is None:
            return None
        try:
            return {k: float(v) for k, v in json.loads(raw).items()}
        except (ValueError, TypeError) as exc:
            logger.warning("corrupt cache entry for %s (%s); recomputing", user_id, exc)
            return None

    def set_cached_rolling(self, user_id: str, rolling: Mapping[str, float]) -> None:
        """Write a user's rolling block with the configured TTL."""
        if not self.redis:
            return
        try:
            self.redis.setex(
                self._key(user_id), self.ttl_seconds, json.dumps(dict(rolling))
            )
        except Exception as exc:  # noqa: BLE001
            # A cache write failure must never fail a recommendation.
            logger.warning("redis write failed (%s); continuing uncached", exc)

    def invalidate(self, user_id: str) -> None:
        """Drop a user's entry — call this when new training history lands."""
        if not self.redis:
            return
        try:
            self.redis.delete(self._key(user_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis delete failed (%s)", exc)

    # -- main entry point -------------------------------------------------

    def build_features(
        self,
        user_id: str,
        signals: Mapping[str, Any],
        history: Optional[Union[pd.DataFrame, Callable[[], pd.DataFrame]]] = None,
    ) -> FeatureFetch:
        """
        Assemble the model input vector for one request.

        Parameters
        ----------
        user_id
            Cache key.
        signals
            Today's raw fields from the request (readiness, hrv, sleep, fatigue,
            profile attributes...).  If the caller already supplies rolling
            fields, they win over both the cache and recomputation — an explicit
            value from the caller is never overridden.
        history
            The user's daily history, or a **callable returning it**.  Needed
            only on a cache miss, so passing a callable keeps the history read
            off the hot path entirely: on a hit it is never invoked.  Passing an
            already-materialised DataFrame means the caller paid for the read
            whether or not the cache needed it, which quietly erases most of the
            measured benefit of caching.

        Returns
        -------
        FeatureFetch
        """
        t0 = time.perf_counter()

        caller_supplied = {f for f in ROLLING_FIELDS if signals.get(f) is not None}
        if len(caller_supplied) == len(ROLLING_FIELDS):
            rolling: Dict[str, float] = {}
            source, cache_hit = "request", False
        else:
            cached = self.get_cached_rolling(user_id) if self.cache_enabled else None
            if cached is not None:
                rolling, source, cache_hit = cached, "redis", True
                self._hits += 1
            else:
                frame = history() if callable(history) else history
                rolling = (
                    compute_rolling_features(frame)
                    if frame is not None and not frame.empty
                    else {}
                )
                source, cache_hit = "recomputed", False
                self._misses += 1
                if rolling:
                    self.set_cached_rolling(user_id, rolling)

        # Caller-supplied values take precedence over cached/recomputed ones.
        merged: Dict[str, Any] = dict(rolling)
        merged.update({k: v for k, v in signals.items() if v is not None})

        features = transform(merged)
        if features.shape != (FEATURE_DIM,):  # pragma: no cover - guard
            raise RuntimeError(
                f"feature vector has shape {features.shape}, expected ({FEATURE_DIM},)"
            )

        return FeatureFetch(
            features=features,
            cache_hit=cache_hit,
            rolling_source=source,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            rolling=dict(rolling),
        )

    # -- introspection ----------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "cache_enabled": bool(self.cache_enabled and self.redis is not None),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "ttl_seconds": self.ttl_seconds,
        }
