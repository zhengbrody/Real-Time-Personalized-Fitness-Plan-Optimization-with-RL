"""
Serving Integration Tests
=========================

Exercises the recommendation path against a **real** Redis instance and through
the actual FastAPI app.

The unit suite already asserts cache parity against an in-memory stand-in.  What
that cannot catch is what happens on the wire: Redis stores strings, so every
cached float makes a round trip through JSON, and a serialisation that loses
precision would produce a warm-cache feature vector that differs from the
cold-cache one by a hair — enough to feed the model inputs it was not trained
on, not enough for anything to look broken.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from src.feature_store.transform import FEATURE_DIM, transform
from src.safety.action_filter import filter_actions
from src.serving.feature_service import ROLLING_FIELDS, FeatureService
from src.serving.history_store import SyntheticHistoryStore


@pytest.fixture
def service(redis_client):
    return FeatureService(redis_client=redis_client, ttl_seconds=60)


@pytest.fixture
def history():
    return SyntheticHistoryStore(days=90).get("itest-user")


def _user_id() -> str:
    """Unique per test so runs cannot contaminate each other."""
    return f"itest-{uuid.uuid4().hex[:12]}"


def _partial(signals: dict) -> dict:
    """Today's signals only — the rolling block must come from cache/history."""
    return {k: v for k, v in signals.items() if k not in ROLLING_FIELDS}


# ---------------------------------------------------------------------------
# Redis round trip
# ---------------------------------------------------------------------------


def test_cold_and_warm_cache_agree_through_real_redis(service, signals, history):
    """
    The assertion that matters: a real serialisation round trip must not change
    a single bit of the feature vector.
    """
    uid = _user_id()
    payload = _partial(signals)

    cold = service.build_features(uid, payload, history=history)
    warm = service.build_features(uid, payload, history=history)

    assert cold.cache_hit is False and cold.rolling_source == "recomputed"
    assert warm.cache_hit is True and warm.rolling_source == "redis"
    assert np.array_equal(cold.features, warm.features)


def test_cached_rolling_values_survive_the_round_trip(service, signals, history):
    """Each individual rolling field must come back exactly as it went in."""
    uid = _user_id()
    fetch = service.build_features(uid, _partial(signals), history=history)
    restored = service.get_cached_rolling(uid)

    assert restored is not None
    for field in ROLLING_FIELDS:
        assert restored[field] == pytest.approx(fetch.rolling[field], rel=0, abs=0)


def test_ttl_is_applied_to_cache_entries(service, redis_client, signals, history):
    """An entry without a TTL would outlive the window it describes."""
    uid = _user_id()
    service.build_features(uid, _partial(signals), history=history)
    ttl = redis_client.ttl(f"profit:features:{uid}")
    assert 0 < ttl <= 60


def test_invalidate_forces_a_recompute(service, signals, history):
    """New training history has to be able to evict a stale entry."""
    uid = _user_id()
    service.build_features(uid, _partial(signals), history=history)
    assert service.build_features(uid, _partial(signals), history=history).cache_hit

    service.invalidate(uid)
    assert (
        service.build_features(uid, _partial(signals), history=history).cache_hit
        is False
    )


def test_history_is_not_read_on_a_cache_hit(service, signals, history):
    """
    The cache exists to avoid the recompute, so a hit must not touch history.
    Passing a callable that raises proves the hit path never invokes it.
    """
    uid = _user_id()
    service.build_features(uid, _partial(signals), history=history)  # warm it

    def explode() -> pd.DataFrame:  # pragma: no cover - must never run
        raise AssertionError("history was read on a cache hit")

    fetch = service.build_features(uid, _partial(signals), history=explode)
    assert fetch.cache_hit is True


def test_uncached_service_matches_the_cached_one(redis_client, signals, history):
    """Turning the cache off changes latency, never the answer."""
    cached = FeatureService(redis_client=redis_client, ttl_seconds=60)
    uncached = FeatureService(redis_client=False)
    payload = _partial(signals)

    a = cached.build_features(_user_id(), payload, history=history).features
    b = uncached.build_features(_user_id(), payload, history=history).features
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# End-to-end through FastAPI
# ---------------------------------------------------------------------------


@pytest.fixture
def client(redis_client, monkeypatch):
    """
    TestClient over the real app.

    The app connects to Redis at import time from the environment, which the
    session fixture has already proven reachable.
    """
    from fastapi.testclient import TestClient

    from src.serving.api_server import app

    return TestClient(app)


def test_recommend_endpoint_returns_a_gated_action(client, signals):
    response = client.post(
        "/recommend_rl", json={"user_id": _user_id(), "signals": signals}
    )
    assert response.status_code == 200

    body = response.json()
    allowed = body["safety"]["allowed_action_ids"]
    assert body["action_id"] in allowed
    assert allowed == filter_actions(signals).allowed_action_ids
    assert body["workout_type"] and body["intensity"]


def test_recommend_respects_the_safety_gate_under_depletion(client, signals):
    """
    A critically depleted state must collapse the action set to rest/recovery —
    end to end, not just in the gate's own unit test.
    """
    depleted = {**signals, "readiness_score": 20.0, "fatigue": 9.5}
    body = client.post(
        "/recommend_rl", json={"user_id": _user_id(), "signals": depleted}
    ).json()

    assert "critical_depletion" in body["safety"]["rules_triggered"]
    assert body["workout_type"] in ("REST", "RECOVERY")
    assert body["intensity"] in ("NONE", "LOW")


def test_second_request_for_a_user_hits_the_cache(client, signals):
    uid = _user_id()
    first = client.post(
        "/recommend_rl", json={"user_id": uid, "signals": signals}
    ).json()
    second = client.post(
        "/recommend_rl", json={"user_id": uid, "signals": signals}
    ).json()

    assert first["features"]["cache_hit"] is False
    assert second["features"]["cache_hit"] is True


def test_feedback_endpoint_updates_the_posterior(client, signals):
    """Feedback must reach the model, not just return 200."""
    from src.serving.api_server import policy_service

    uid = _user_id()
    rec = client.post("/recommend_rl", json={"user_id": uid, "signals": signals}).json()
    before = int(np.sum(policy_service.model.action_counts))

    response = client.post(
        "/feedback_rl",
        json={
            "user_id": uid,
            "action_id": rec["action_id"],
            "reward": 0.8,
            "signals": signals,
        },
    )
    assert response.status_code == 200
    assert int(np.sum(policy_service.model.action_counts)) == before + 1


def test_feedback_rejects_an_out_of_range_reward(client, signals):
    response = client.post(
        "/feedback_rl",
        json={
            "user_id": _user_id(),
            "action_id": 0,
            "reward": 42.0,
            "signals": signals,
        },
    )
    assert response.status_code == 422


def test_health_reports_the_cache_as_enabled(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["feature_cache"]["cache_enabled"] is True


def test_served_features_are_exactly_the_shared_transform(client, signals):
    """
    Training–serving parity through the running service.

    The service does two things: fill the rolling block (from Redis or a
    recompute) and merge it under the request's own fields.  Everything after
    that must be the shared transform and nothing else — so re-running the
    transform on the merged record has to reproduce the served vector exactly.
    Any serving-only adjustment, however small, breaks this.
    """
    uid = _user_id()
    client.post("/recommend_rl", json={"user_id": uid, "signals": signals})

    from src.serving.api_server import feature_service, history_store

    fetch = feature_service.build_features(
        uid, signals, history=lambda: history_store.get(uid)
    )
    assert fetch.features.shape == (FEATURE_DIM,)

    merged = {**fetch.rolling, **signals}
    assert np.array_equal(fetch.features, transform(merged))
