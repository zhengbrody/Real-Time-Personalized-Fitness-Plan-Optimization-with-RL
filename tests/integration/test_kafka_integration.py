"""
Kafka Online-Learning Integration Tests
=======================================

Verifies the feedback loop against a **real** broker: recommendation ->
feedback event -> consumer -> posterior update.

The failure mode worth testing for is not "Kafka is down" — that is loud.  It is
a loop that appears healthy while doing nothing: events flow, the consumer logs
them, and the model never moves.  So these tests assert on the *model*, not on
delivery, and they check that the features the consumer reconstructs match the
ones the decision was made from.
"""

from __future__ import annotations

import json
import time
import uuid

import numpy as np
import pytest

from src.feature_store.transform import transform
from src.recommendation.action_space import ActionSpace
from src.safety.action_filter import filter_actions
from src.serving.policy_service import PolicyService

CONSUME_TIMEOUT_MS = 20_000


@pytest.fixture
def topic(kafka_bootstrap):
    """A fresh topic per test, cleaned up afterwards."""
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    name = f"itest.feedback.{uuid.uuid4().hex[:12]}"
    admin = KafkaAdminClient(bootstrap_servers=kafka_bootstrap)
    try:
        admin.create_topics(
            [NewTopic(name=name, num_partitions=1, replication_factor=1)]
        )
    except TopicAlreadyExistsError:  # pragma: no cover
        pass

    yield name

    try:
        admin.delete_topics([name])
    except Exception:  # noqa: BLE001 - cleanup is best effort
        pass
    finally:
        admin.close()


def _producer(bootstrap):
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,
    )


def _consumer(bootstrap, topic):
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=f"itest-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=CONSUME_TIMEOUT_MS,
    )
    # Force group join so nothing produced afterwards is missed.
    consumer.poll(timeout_ms=5000)
    return consumer


def _feedback_event(signals, action_id, reward):
    return {
        "user_id": "itest-user",
        "action_id": int(action_id),
        "reward": float(reward),
        "signals": signals,
        "features": transform(signals).tolist(),
        "emitted_at": time.time(),
    }


def test_feedback_events_round_trip_through_the_broker(kafka_bootstrap, topic, signals):
    consumer = _consumer(kafka_bootstrap, topic)
    producer = _producer(kafka_bootstrap)

    sent = [
        _feedback_event(signals, aid, 0.5 + 0.1 * i) for i, aid in enumerate([3, 5, 9])
    ]
    for event in sent:
        producer.send(topic, event)
    producer.flush()
    producer.close()

    received = []
    for message in consumer:
        received.append(message.value)
        if len(received) == len(sent):
            break
    consumer.close()

    assert len(received) == len(sent)
    assert [e["action_id"] for e in received] == [e["action_id"] for e in sent]
    assert [e["reward"] for e in received] == pytest.approx([e["reward"] for e in sent])


def test_consumed_events_move_the_posterior(kafka_bootstrap, topic, signals):
    """
    The check that catches a loop which is plumbed but never closes: posterior
    counts must rise by exactly the number of events consumed.
    """
    policy = PolicyService()
    before = int(np.sum(policy.model.action_counts))

    consumer = _consumer(kafka_bootstrap, topic)
    producer = _producer(kafka_bootstrap)

    n_events = 5
    for i in range(n_events):
        producer.send(topic, _feedback_event(signals, 3 + i, 0.6))
    producer.flush()
    producer.close()

    consumed = 0
    for message in consumer:
        event = message.value
        policy.update(transform(event["signals"]), event["action_id"], event["reward"])
        consumed += 1
        if consumed == n_events:
            break
    consumer.close()

    assert consumed == n_events
    assert int(np.sum(policy.model.action_counts)) == before + n_events


def test_consumer_side_features_match_the_producer_side(
    kafka_bootstrap, topic, signals
):
    """
    The quiet failure: the loop closes, but the consumer rebuilds a different
    feature vector than the recommendation was made from, so the posterior
    learns against inputs the policy never saw.
    """
    consumer = _consumer(kafka_bootstrap, topic)
    producer = _producer(kafka_bootstrap)

    event = _feedback_event(signals, 7, 0.75)
    producer.send(topic, event)
    producer.flush()
    producer.close()

    received = next(iter(consumer))
    consumer.close()

    rebuilt = transform(received.value["signals"])
    assert np.array_equal(rebuilt, np.array(received.value["features"]))


def test_full_loop_recommendation_to_posterior_update(kafka_bootstrap, topic, signals):
    """
    End to end: gate -> policy -> event -> broker -> consumer -> posterior, with
    the served action asserted to be inside the gated set at both ends.
    """
    space = ActionSpace()
    policy = PolicyService()

    features = transform(signals)
    allowed = filter_actions(signals, space).allowed_action_ids
    action_id = int(policy.model.select_action(features, allowed))
    assert action_id in allowed

    before = int(policy.model.action_counts[action_id])

    producer = _producer(kafka_bootstrap)
    consumer = _consumer(kafka_bootstrap, topic)
    producer.send(topic, _feedback_event(signals, action_id, 0.9))
    producer.flush()
    producer.close()

    message = next(iter(consumer))
    consumer.close()

    event = message.value
    assert event["action_id"] in allowed
    policy.update(transform(event["signals"]), event["action_id"], event["reward"])

    assert int(policy.model.action_counts[action_id]) == before + 1


def test_loop_latency_is_bounded(kafka_bootstrap, topic, signals):
    """
    A loose ceiling, not a benchmark. It exists so that a regression turning a
    millisecond loop into a multi-second one fails the build instead of being
    noticed later.
    """
    consumer = _consumer(kafka_bootstrap, topic)
    producer = _producer(kafka_bootstrap)

    producer.send(topic, _feedback_event(signals, 4, 0.5))
    producer.flush()
    producer.close()

    message = next(iter(consumer))
    consumer.close()

    latency_ms = (time.time() - message.value["emitted_at"]) * 1000.0
    assert 0 <= latency_ms < 10_000
