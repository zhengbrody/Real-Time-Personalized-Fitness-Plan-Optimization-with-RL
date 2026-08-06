"""
Training–Serving Parity
=======================

These tests are the enforcement mechanism behind the claim that the offline and
online paths cannot drift apart.

Training–serving skew is not usually introduced by someone rewriting the
feature logic on purpose.  It arrives when the serving path grows a "small"
convenience — a different default for a missing field, a cache that rounds a
float, a re-ordered column list — and nothing fails, because nothing was
checking.  The model keeps returning plausible recommendations computed from
subtly different inputs than it was trained on.

So each test below pins one specific way the two paths could diverge:

- the feature schema itself (order, uniqueness, dimension)
- the transform's purity and determinism
- offline vs. online assembly of the same record
- a warm cache vs. a cold cache
- the safety gate as seen by the benchmark vs. by the serving policy
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from src.feature_store.transform import (
    FEATURE_DIM,
    FEATURE_NAMES,
    describe,
    transform,
)
from src.recommendation.action_space import ActionSpace
from src.safety.action_filter import filter_actions
from src.serving.feature_service import (
    ROLLING_FIELDS,
    FeatureService,
    compute_rolling_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeRedis:
    """
    Minimal in-memory stand-in for the Redis client.

    Deliberately not a mock: it round-trips through real JSON serialisation, so
    a value that would lose precision on the way into the cache loses it here
    too.  A mock that returned the original Python object would hide exactly
    the class of bug the cache-parity test exists to catch.
    """

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


@pytest.fixture
def raw_state() -> Dict[str, Any]:
    """A complete raw body-state record, as the wearable pipeline emits it."""
    return {
        "readiness_score": 68.0,
        "sleep_score": 74.0,
        "sleep_duration_hours": 7.1,
        "sleep_debt_7d": 2.3,
        "hrv": 54.0,
        "hrv_7d_mean": 51.2,
        "hrv_28d_mean": 49.8,
        "hrv_28d_std": 6.4,
        "resting_hr": 58.0,
        "resting_hr_baseline": 57.0,
        "fatigue": 4.0,
        "activity_score": 66.0,
        "acwr": 1.12,
        "load_7d": 940.0,
        "load_28d_mean": 210.0,
        "days_since_training": 1.0,
        "consecutive_hard_days": 1.0,
        "completion_rate_7d": 0.71,
        "completion_rate_28d": 0.68,
        "streak": 3.0,
        "sessions_7d": 4.0,
        "days_active_28d": 17.0,
        "day_of_week": 3.0,
        "week_phase": 0.42,
        "training_age_years": 4.5,
        "age": 31.0,
        "goal": "strength",
        "intensity_tolerance": 0.62,
        "bodyweight_kg": 78.0,
    }


@pytest.fixture
def history() -> pd.DataFrame:
    """90 days of deterministic training history for the cache-miss path."""
    rng = np.random.default_rng(7)
    n = 90
    return pd.DataFrame(
        {
            "hrv": rng.normal(52, 6, n),
            "resting_hr": rng.normal(58, 3, n),
            "sleep_hours": rng.normal(7.2, 0.7, n),
            "load": rng.choice([0.0, 90.0, 180.0, 270.0], n),
            "completed": rng.integers(0, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_feature_names_match_dimension():
    assert len(FEATURE_NAMES) == FEATURE_DIM


def test_feature_names_are_unique():
    """A duplicated name silently makes one column unreachable via describe()."""
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_transform_output_shape_and_order(raw_state):
    vector = transform(raw_state)
    assert vector.shape == (FEATURE_DIM,)
    assert vector.dtype == np.float64
    # describe() must invert the vector back onto the schema in the same order.
    named = describe(vector)
    assert list(named.keys()) == FEATURE_NAMES


def test_transform_is_finite_and_bounded(raw_state):
    vector = transform(raw_state)
    assert np.all(np.isfinite(vector))
    # Everything is normalised into roughly [-1, 1]; a value far outside that
    # means a normalisation constant is wrong for this field.
    assert np.all(np.abs(vector) <= 1.5)


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_transform_is_deterministic(raw_state):
    assert np.array_equal(transform(raw_state), transform(raw_state))


def test_transform_does_not_mutate_input(raw_state):
    before = json.dumps(raw_state, sort_keys=True)
    transform(raw_state)
    assert json.dumps(raw_state, sort_keys=True) == before


def test_transform_handles_missing_fields_via_indicators():
    """A partial record produces a valid vector and raises the missing flags."""
    vector = transform({"readiness_score": 60.0, "fatigue": 5.0})
    named = describe(vector)
    assert np.all(np.isfinite(vector))
    assert named["hrv_missing"] == 1.0
    assert named["sleep_missing"] == 1.0


def test_transform_missing_indicators_clear_when_present(raw_state):
    named = describe(transform(raw_state))
    assert named["hrv_missing"] == 0.0
    assert named["sleep_missing"] == 0.0


# ---------------------------------------------------------------------------
# Offline vs. online
# ---------------------------------------------------------------------------


def test_offline_and_online_paths_agree(raw_state):
    """
    The core parity assertion.

    Offline (benchmark/training) calls ``transform`` on the raw record.  Online
    calls ``FeatureService.build_features``, which merges cache/rolling data and
    then calls ``transform``.  Given a record that already carries every rolling
    field, the two must produce bit-identical vectors — not merely close.
    """
    offline = transform(raw_state)

    service = FeatureService(redis_client=FakeRedis())
    online = service.build_features("user-1", raw_state).features

    assert np.array_equal(offline, online), (
        "offline and online feature vectors differ at indices "
        f"{np.flatnonzero(offline != online).tolist()}"
    )


def test_caller_supplied_rolling_fields_are_never_overridden(raw_state, history):
    """
    An explicit value in the request must win over a cached one.

    Otherwise a backfill or a replay that supplies known-good history would be
    silently overwritten by whatever happens to be in the cache.
    """
    service = FeatureService(redis_client=FakeRedis())
    service.set_cached_rolling("user-1", {f: 0.0 for f in ROLLING_FIELDS})

    online = service.build_features("user-1", raw_state).features
    assert np.array_equal(transform(raw_state), online)


# ---------------------------------------------------------------------------
# Cache parity
# ---------------------------------------------------------------------------


def _partial_signals(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    """Today's signals only — the rolling block must come from cache/history."""
    return {k: v for k, v in raw_state.items() if k not in ROLLING_FIELDS}


def test_cold_and_warm_cache_produce_identical_features(raw_state, history):
    """
    A cache hit and a cache miss must be indistinguishable in the output.

    This is the test that catches a cache which round-trips floats through a
    lossy encoding: the miss path computes in float64, the hit path reads back
    whatever survived serialisation, and only a bit-exact comparison notices.
    """
    service = FeatureService(redis_client=FakeRedis())
    signals = _partial_signals(raw_state)

    cold = service.build_features("user-1", signals, history=history)
    warm = service.build_features("user-1", signals, history=history)

    assert cold.cache_hit is False
    assert cold.rolling_source == "recomputed"
    assert warm.cache_hit is True
    assert warm.rolling_source == "redis"
    assert np.array_equal(cold.features, warm.features)


def test_cache_disabled_matches_cache_enabled(raw_state, history):
    """Turning the cache off changes latency, never the answer."""
    cached = FeatureService(redis_client=FakeRedis())
    uncached = FeatureService(redis_client=False)
    signals = _partial_signals(raw_state)

    a = cached.build_features("user-1", signals, history=history).features
    b = uncached.build_features("user-1", signals, history=history).features
    assert np.array_equal(a, b)


def test_unreachable_redis_degrades_without_changing_output(raw_state, history):
    """A Redis outage must not change what the model is fed."""

    class BrokenRedis(FakeRedis):
        def get(self, key: str):
            raise ConnectionError("redis is down")

        def setex(self, key: str, ttl: int, value: str) -> None:
            raise ConnectionError("redis is down")

    signals = _partial_signals(raw_state)
    healthy = FeatureService(redis_client=FakeRedis())
    broken = FeatureService(redis_client=BrokenRedis())

    a = healthy.build_features("user-1", signals, history=history).features
    b = broken.build_features("user-1", signals, history=history).features
    assert np.array_equal(a, b)


def test_corrupt_cache_entry_falls_back_to_recompute(raw_state, history):
    """A malformed cache value must be ignored, not propagated."""
    fake = FakeRedis()
    fake.store["profit:features:user-1"] = "{not valid json"

    service = FeatureService(redis_client=fake)
    fetch = service.build_features(
        "user-1", _partial_signals(raw_state), history=history
    )

    assert fetch.rolling_source == "recomputed"
    assert np.all(np.isfinite(fetch.features))


def test_rolling_features_cover_every_declared_field(history):
    rolling = compute_rolling_features(history)
    assert set(rolling.keys()) == set(ROLLING_FIELDS)
    assert all(np.isfinite(v) for v in rolling.values())


# ---------------------------------------------------------------------------
# Safety-gate parity
# ---------------------------------------------------------------------------


def test_safety_gate_is_shared_between_paths(raw_state):
    """
    The benchmark and the serving policy must gate on the same rules.

    They previously did not: the benchmark carried a private copy of the filter
    that omitted the consecutive-hard-days rule, so the policy was measured
    under weaker constraints than it runs under.
    """
    from src.serving.policy_service import PolicyService

    space = ActionSpace()
    offline_allowed = filter_actions(raw_state, space).allowed_action_ids

    service = PolicyService()
    decision = service.recommend(raw_state, explore=False)

    assert decision.allowed_action_ids == offline_allowed
    assert decision.action_id in offline_allowed


@pytest.mark.parametrize(
    "state,expected_rule",
    [
        ({"readiness_score": 20.0, "fatigue": 5.0}, "critical_depletion"),
        ({"readiness_score": 70.0, "fatigue": 9.0}, "critical_depletion"),
        ({"readiness_score": 70.0, "fatigue": 7.0}, "elevated_fatigue"),
        (
            {"readiness_score": 70.0, "fatigue": 3.0, "consecutive_hard_days": 4.0},
            "consecutive_high_load",
        ),
        ({"readiness_score": 70.0, "fatigue": 3.0, "acwr": 1.8}, "acute_chronic_spike"),
        (
            {"readiness_score": 70.0, "fatigue": 3.0, "sleep_duration_hours": 3.0},
            "low_hrv_or_sleep",
        ),
    ],
)
def test_safety_rules_fire_as_documented(state, expected_rule):
    decision = filter_actions(state)
    assert expected_rule in decision.triggered_rules


def test_safety_gate_never_returns_an_empty_action_set():
    """
    Even in the worst state the policy must have a legal action.

    An empty allowed set would make the bandit's argmax undefined; failing
    closed to REST is the only safe degenerate answer.
    """
    decision = filter_actions(
        {
            "readiness_score": 1.0,
            "fatigue": 10.0,
            "sleep_duration_hours": 0.0,
            "hrv": 1.0,
            "soreness": 10.0,
            "acwr": 5.0,
            "consecutive_hard_days": 5.0,
        }
    )
    assert decision.allowed_action_ids
    assert 0 in decision.allowed_action_ids


def test_missing_fields_fail_closed_not_open():
    """
    An empty record must be treated as conservatively as a bad one.

    The dangerous failure mode is the opposite: defaulting missing fatigue to
    zero would let a recommender get *bolder* the less it knows about the user.
    """
    empty = filter_actions({})
    fresh = filter_actions(
        {
            "readiness_score": 85.0,
            "fatigue": 2.0,
            "sleep_duration_hours": 8.0,
            "hrv": 60.0,
        }
    )
    assert len(empty.allowed_action_ids) < len(fresh.allowed_action_ids)
    assert empty.is_restricted
