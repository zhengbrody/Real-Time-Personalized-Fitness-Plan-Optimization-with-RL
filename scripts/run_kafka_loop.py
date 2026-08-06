"""
Kafka Online-Learning Closed Loop
=================================

Runs the full feedback loop end to end over a real Kafka broker and measures it:

    features -> safety gate -> Thompson Sampling -> simulated user response
      -> feedback event published to Kafka
      -> consumer reads it, rebuilds features, updates the posterior
      -> the next recommendation reflects the update

Why measure it rather than describe it
--------------------------------------
"Kafka-based online learning" is easy to claim and easy to get subtly wrong.
The two failure modes that matter are silent:

1. **The loop is plumbed but never closes.**  Events flow, the consumer logs
   them, and nothing reaches the model.  Everything looks healthy.
2. **The loop closes but the features drift.**  The consumer rebuilds the
   feature vector from the event payload, and it does not match the vector the
   recommendation was made from, so the posterior learns against inputs the
   policy never saw.

So this script asserts the loop actually closes — posterior counts must rise by
exactly the number of consumed events — and it re-derives features on the
consumer side through the *same* shared transform, then verifies they match the
vector recorded at decision time.

It also reports loop latency: the wall time from publishing feedback to the
model having absorbed it, which is what bounds how quickly the system can react
to a user.

Usage
-----
    python scripts/run_kafka_loop.py --events 2000
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_store.transform import transform  # noqa: E402
from src.recommendation.action_space import ActionSpace  # noqa: E402
from src.safety.action_filter import filter_actions  # noqa: E402
from src.serving.policy_service import PolicyService  # noqa: E402
from src.simulation.fitness_env import FitnessSimulator  # noqa: E402

TOPIC = "training.user.feedback"


def ensure_topic(bootstrap: str, topic: str) -> None:
    """Create the topic if it does not exist."""
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = KafkaAdminClient(bootstrap_servers=bootstrap)
    try:
        admin.create_topics(
            [NewTopic(name=topic, num_partitions=1, replication_factor=1)]
        )
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


class FeedbackConsumerThread(threading.Thread):
    """
    Consumes feedback events and folds them into the live policy.

    Runs in a thread rather than a separate process so the test can inspect the
    posterior directly and prove the update landed.  In deployment this is
    ``src/online_learning/kafka_consumer.py`` as its own service; the update
    path it exercises is identical.
    """

    def __init__(
        self,
        policy: PolicyService,
        bootstrap: str,
        topic: str,
        expected: int,
        results: queue.Queue,
    ):
        super().__init__(daemon=True)
        self.policy = policy
        self.bootstrap = bootstrap
        self.topic = topic
        self.expected = expected
        self.results = results
        self.consumed = 0
        self.feature_mismatches = 0
        self.loop_latencies: List[float] = []
        self.first_consume_at: Optional[float] = None
        self.last_consume_at: Optional[float] = None
        # NB: not `self._stop` — that name shadows Thread's internal method.
        self._stop_flag = threading.Event()
        # Signals that the consumer has joined the group and been assigned its
        # partitions. Producing before that point means the first few hundred
        # events sit in the topic waiting for a consumer that does not exist
        # yet, and their queue wait shows up as loop latency — measuring
        # consumer startup rather than the loop.
        self.ready = threading.Event()

    def run(self) -> None:  # noqa: D102
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap,
            group_id=f"online-learning-{int(time.time())}",
            auto_offset_reset="earliest",
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            consumer_timeout_ms=20_000,
            enable_auto_commit=True,
        )

        # Force the group join before signalling readiness. The topic is empty
        # at this point, so this poll returns nothing and drops no events.
        consumer.poll(timeout_ms=5000)
        self.ready.set()

        for message in consumer:
            if self._stop_flag.is_set():
                break
            event = message.value

            # Rebuild the feature vector on the consumer side, through the same
            # shared transform the recommendation used. Recomputing rather than
            # trusting a serialised vector is what keeps the update honest —
            # and lets the mismatch check below actually mean something.
            features = transform(event["signals"])
            if not np.allclose(features, np.array(event["features"]), rtol=0, atol=0):
                self.feature_mismatches += 1

            self.policy.update(features, event["action_id"], event["reward"])

            now = time.time()
            if self.first_consume_at is None:
                self.first_consume_at = now
            self.last_consume_at = now

            self.consumed += 1
            self.loop_latencies.append((now - event["emitted_at"]) * 1000.0)

            if self.consumed >= self.expected:
                break

        consumer.close()
        drain_seconds = (
            (self.last_consume_at - self.first_consume_at)
            if self.first_consume_at and self.last_consume_at
            else 0.0
        )
        self.results.put(
            {
                "consumed": self.consumed,
                "feature_mismatches": self.feature_mismatches,
                "loop_latencies_ms": self.loop_latencies,
                "drain_seconds": drain_seconds,
                "drain_rate": (
                    (self.consumed / drain_seconds) if drain_seconds > 0 else 0.0
                ),
            }
        )

    def stop(self) -> None:
        self._stop_flag.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kafka online-learning closed loop")
    parser.add_argument("--events", type=int, default=2000)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help=(
            "target events/s. Pacing matters: an unpaced producer outruns the "
            "consumer, and the resulting queue wait dominates the measured loop "
            "latency, so the number would describe the burst rather than the "
            "system. Pass 0 to measure maximum drain throughput instead."
        ),
    )
    args = parser.parse_args()

    from kafka import KafkaProducer

    print("=" * 88)
    print("  Kafka Online-Learning Closed Loop")
    print("=" * 88)
    print(
        f"\n  Broker: {args.bootstrap}   events: {args.events}   users: {args.users}\n"
    )

    topic = f"{TOPIC}.{int(time.time())}"
    ensure_topic(args.bootstrap, topic)

    space = ActionSpace()
    policy = PolicyService()
    print(f"  policy: {policy.model_name}")

    counts_before = int(np.sum(policy.model.action_counts))

    results_q: queue.Queue = queue.Queue()
    consumer = FeedbackConsumerThread(
        policy, args.bootstrap, topic, args.events, results_q
    )
    consumer.start()
    if not consumer.ready.wait(timeout=60):
        raise SystemExit("consumer failed to join the group within 60s")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,
    )

    sim = FitnessSimulator(action_space=space, seed=args.seed)
    users = [sim.new_user() for _ in range(args.users)]

    rewards: List[float] = []
    publish_ms: List[float] = []

    print("  producing ...", end="", flush=True)
    t_start = time.time()
    for i in range(args.events):
        user = users[i % args.users]
        raw = sim.observe(user)
        # Only the JSON-serialisable signal fields go on the wire.
        signals = {k: v for k, v in raw.items() if isinstance(v, (int, float, str))}
        features = transform(signals)

        allowed = filter_actions(signals, space).allowed_action_ids
        action_id = int(policy.model.select_action(features, allowed))
        reward = sim.step(user, raw, action_id)
        rewards.append(reward)

        t0 = time.perf_counter()
        producer.send(
            topic,
            {
                "user_id": f"u{user.profile.user_id}",
                "action_id": action_id,
                "reward": float(reward),
                "signals": signals,
                "features": features.tolist(),
                "emitted_at": time.time(),
            },
        )
        publish_ms.append((time.perf_counter() - t0) * 1000.0)

        if args.rate > 0:
            target = t_start + (i + 1) / args.rate
            sleep_for = target - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

    producer.flush()
    producer.close()
    produce_elapsed = time.time() - t_start
    print(f" done ({produce_elapsed:.1f}s)")

    print("  draining consumer ...", end="", flush=True)
    consumer.join(timeout=120)
    consumer.stop()
    try:
        stats = results_q.get(timeout=10)
    except queue.Empty:
        stats = {
            "consumed": consumer.consumed,
            "feature_mismatches": -1,
            "loop_latencies_ms": [],
        }
    print(" done")

    counts_after = int(np.sum(policy.model.action_counts))
    applied = counts_after - counts_before

    lat = np.array(stats["loop_latencies_ms"], dtype=float)
    n = len(rewards)
    q = max(1, n // 4)

    print()
    print("  RESULTS")
    print("  " + "-" * 84)
    print(f"  events produced           : {args.events}")
    print(f"  events consumed           : {stats['consumed']}")
    print(f"  posterior updates applied : {applied}")
    print(f"  feature mismatches        : {stats['feature_mismatches']}")
    if len(lat):
        print(
            f"  loop latency (publish -> model updated): "
            f"p50 {np.percentile(lat, 50):.1f} ms  "
            f"p95 {np.percentile(lat, 95):.1f} ms  "
            f"p99 {np.percentile(lat, 99):.1f} ms"
        )
    print(
        f"  producer send (async enqueue)         : p50 {np.percentile(publish_ms, 50):.3f} ms"
    )
    print(
        f"  producer rate                         : "
        f"{args.events / produce_elapsed:.0f} events/s"
        + (f" (paced to {args.rate:.0f})" if args.rate > 0 else " (unpaced)")
    )
    print(
        f"  consumer drain rate                   : "
        f"{stats.get('drain_rate', 0.0):.0f} events/s"
    )
    print()
    print(
        f"  reward first quarter {np.mean(rewards[:q]):.4f}  ->  "
        f"last quarter {np.mean(rewards[-q:]):.4f}  "
        f"({(np.mean(rewards[-q:]) - np.mean(rewards[:q])) / abs(np.mean(rewards[:q])) * 100:+.1f}%)"
    )

    # The loop is only closed if every consumed event actually moved the model.
    loop_closed = applied == stats["consumed"] and stats["consumed"] > 0
    features_consistent = stats["feature_mismatches"] == 0
    print()
    print(f"  loop closed               : {'PASS' if loop_closed else 'FAIL'}")
    print(f"  producer/consumer features: {'PASS' if features_consistent else 'FAIL'}")

    payload = {
        "config": {
            "events": args.events,
            "users": args.users,
            "seed": args.seed,
            "broker": args.bootstrap,
            "topic": topic,
        },
        "results": {
            "produced": args.events,
            "consumed": stats["consumed"],
            "posterior_updates_applied": applied,
            "feature_mismatches": stats["feature_mismatches"],
            "loop_latency_ms": {
                "p50": float(np.percentile(lat, 50)) if len(lat) else None,
                "p95": float(np.percentile(lat, 95)) if len(lat) else None,
                "p99": float(np.percentile(lat, 99)) if len(lat) else None,
                "mean": float(lat.mean()) if len(lat) else None,
            },
            "throughput_events_per_s": args.events / produce_elapsed,
            "reward_first_quarter": float(np.mean(rewards[:q])),
            "reward_last_quarter": float(np.mean(rewards[-q:])),
            "loop_closed": loop_closed,
            "features_consistent": features_consistent,
        },
    }
    out = Path(__file__).parent / "kafka_loop_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  Raw results -> {out}")

    if not (loop_closed and features_consistent):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
