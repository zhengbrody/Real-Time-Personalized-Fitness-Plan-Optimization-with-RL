"""
Integration-test fixtures.

These tests talk to **real** Redis and Kafka rather than fakes.  The unit suite
already covers the logic against an in-memory stand-in; what is left to verify
is the part a fake cannot tell you — that the serialisation survives a real
round trip, that a real broker delivers what a real consumer folds into the
model, and that the service degrades correctly when a dependency is missing.

Every fixture skips rather than fails when its dependency is unreachable, so
``pytest tests/`` still works on a laptop with nothing running.  CI provisions
both services, so there the tests actually execute.
"""

from __future__ import annotations

import os

import pytest

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture(scope="session")
def redis_client():
    """A live Redis connection, or skip."""
    redis = pytest.importorskip("redis", reason="redis package not installed")
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        client.ping()
    except Exception as exc:  # noqa: BLE001 - any failure means "no server"
        pytest.skip(f"Redis unreachable at {REDIS_HOST}:{REDIS_PORT} ({exc})")
    yield client
    # Leave no test keys behind for the next run.
    for key in client.scan_iter("profit:features:itest-*"):
        client.delete(key)


@pytest.fixture(scope="session")
def kafka_bootstrap():
    """Bootstrap servers for a reachable broker, or skip."""
    pytest.importorskip("kafka", reason="kafka client not installed")
    from kafka.admin import KafkaAdminClient

    try:
        admin = KafkaAdminClient(
            bootstrap_servers=KAFKA_BOOTSTRAP, request_timeout_ms=5000
        )
        admin.describe_cluster()
        admin.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Kafka unreachable at {KAFKA_BOOTSTRAP} ({exc})")
    return KAFKA_BOOTSTRAP


@pytest.fixture
def signals():
    """A well-formed request payload with no safety rules triggered."""
    return {
        "readiness_score": 72.0,
        "sleep_score": 78.0,
        "sleep_duration_hours": 7.4,
        "hrv": 55.0,
        "resting_hr": 57.0,
        "fatigue": 3.0,
        "activity_score": 64.0,
        "day_of_week": 2.0,
        "age": 30.0,
        "bodyweight_kg": 76.0,
        "training_age_years": 4.0,
        "goal": "strength",
        "intensity_tolerance": 0.6,
    }
