"""
API Latency Benchmark
=====================

Measures serving latency over **real HTTP** against a uvicorn server, not
through FastAPI's in-process ``TestClient``.

Why the change
--------------
The previous version of this script used ``TestClient``, which calls the ASGI
app directly in the same process.  It reported sub-millisecond p99 — a number
that is real but answers a question nobody asked: it excludes the event loop,
the socket, JSON serialisation over the wire, and connection handling, which is
most of what a client actually waits for.  Quoting an in-process figure as a
service latency overstates it by an order of magnitude.

So this script starts a real server, talks to it over a real socket, and reports
what a client would see.  The in-process figure is still measured, and reported
alongside, because the *gap* between the two is the useful part: it separates
model computation from transport.

Arms
----
``/recommend_rl`` cold
    A fresh user id on every request, so the feature cache always misses and the
    rolling block is recomputed from ~90 days of history.

``/recommend_rl`` warm
    A pre-warmed pool of user ids, so the rolling block is served from Redis.
    The difference between this and the cold arm is what the cache buys.

``/health``
    Transport floor: how much of the latency is the HTTP stack alone.

Usage
-----
    python scripts/benchmark_latency.py --n 2000 --port 8123
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

# httpx logs an INFO line per request. At two thousand requests that logging is
# a measurable share of the thing being measured, so it is silenced here rather
# than left to quietly inflate every number in the table.
import logging  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def percentiles(samples: List[float]) -> Dict[str, float]:
    """Latency summary. Percentiles, not just the mean — the mean hides the tail."""
    arr = np.array(samples, dtype=float)
    return {
        "n": len(arr),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "stdev_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "p50_ms": float(np.percentile(arr, 50)),
        "p75_ms": float(np.percentile(arr, 75)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "qps_serial": 1000.0 / float(arr.mean()) if arr.mean() > 0 else float("inf"),
    }


def _signals(i: int) -> dict:
    """A realistic request payload; varied so nothing is trivially memoised."""
    return {
        "readiness_score": 55 + (i % 35),
        "sleep_score": 65 + (i % 25),
        "sleep_duration_hours": 6.5 + (i % 5) * 0.3,
        "hrv": 45 + (i % 25),
        "resting_hr": 54 + (i % 12),
        "fatigue": 2 + (i % 6),
        "activity_score": 50 + (i % 40),
        "day_of_week": i % 7,
        "age": 25 + (i % 30),
        "bodyweight_kg": 60 + (i % 40),
        "training_age_years": 1 + (i % 8),
        "goal": ["strength", "endurance", "general"][i % 3],
        "intensity_tolerance": 0.3 + (i % 6) * 0.1,
    }


def measure(fn: Callable[[int], None], n: int, warmup: int) -> List[float]:
    """Time ``n`` calls, discarding ``warmup`` first to skip JIT/connection setup."""
    for i in range(warmup):
        fn(i)
    samples: List[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(warmup + i)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start_server(port: int, env: dict) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.serving.api_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_for_health(base: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 - server not up yet
            pass
        time.sleep(0.4)
    return False


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_http_arms(base: str, n: int, warmup: int) -> Dict[str, dict]:
    results: Dict[str, dict] = {}

    with httpx.Client(base_url=base, timeout=30.0) as client:
        # -- transport floor ------------------------------------------------
        results["GET /health"] = percentiles(
            measure(lambda i: client.get("/health").raise_for_status(), n // 4, warmup)
        )

        # -- cold cache: fresh user id every call ---------------------------
        run_id = int(time.time())

        def cold(i: int) -> None:
            client.post(
                "/recommend_rl",
                json={"user_id": f"cold-{run_id}-{i}", "signals": _signals(i)},
            ).raise_for_status()

        results["POST /recommend_rl (cache miss)"] = percentiles(
            measure(cold, n, warmup)
        )

        # -- warm cache: small pool, pre-warmed ------------------------------
        pool = [f"warm-{run_id}-{k}" for k in range(20)]
        for k, uid in enumerate(pool):
            client.post(
                "/recommend_rl", json={"user_id": uid, "signals": _signals(k)}
            ).raise_for_status()

        def warm(i: int) -> None:
            client.post(
                "/recommend_rl",
                json={"user_id": pool[i % len(pool)], "signals": _signals(i)},
            ).raise_for_status()

        results["POST /recommend_rl (cache hit)"] = percentiles(
            measure(warm, n, warmup)
        )

        # Confirm the warm arm really did hit — a silently-cold "warm" arm
        # would make the cache look useless and nothing else would notice.
        probe = client.post(
            "/recommend_rl", json={"user_id": pool[0], "signals": _signals(0)}
        ).json()
        results["POST /recommend_rl (cache hit)"]["verified_cache_hit"] = bool(
            probe.get("features", {}).get("cache_hit")
        )

    return results


def run_compute_arm(n: int, warmup: int, redis_port: int) -> Dict[str, dict]:
    """
    Pure compute path: feature assembly + safety gate + posterior sampling, with
    no HTTP layer at all.

    Not measured through ``TestClient``: that still runs an ASGI stack, an httpx
    transport and JSON codecs, so it measures the test harness as much as the
    model.  Calling the services directly is the only way to isolate what the
    recommendation itself costs, and subtracting it from the end-to-end HTTP
    number gives the real transport overhead.
    """
    os.environ["REDIS_PORT"] = str(redis_port)

    from src.serving.feature_service import FeatureService
    from src.serving.history_store import SyntheticHistoryStore
    from src.serving.policy_service import PolicyService

    features = FeatureService()
    policy = PolicyService()
    history = SyntheticHistoryStore()

    run_id = int(time.time())
    pool = [f"compute-{run_id}-{k}" for k in range(20)]
    for k, uid in enumerate(pool):
        features.build_features(uid, _signals(k), history=lambda u=uid: history.get(u))

    def call(i: int) -> None:
        uid = pool[i % len(pool)]
        fetch = features.build_features(
            uid, _signals(i), history=lambda: history.get(uid)
        )
        policy.recommend(_signals(i), features=fetch.features)

    return {"compute only (no HTTP, cache hit)": percentiles(measure(call, n, warmup))}


def run_concurrent_arm(base: str, n: int, concurrency: int) -> Dict[str, dict]:
    """
    Throughput under concurrent clients.

    The serial QPS reported for the other arms is ``1000 / mean_latency`` — the
    rate a *single* client can drive, which understates what the service can
    actually sustain, since most of a request is spent waiting rather than
    computing.  This arm measures achieved throughput with several clients in
    flight.
    """
    from concurrent.futures import ThreadPoolExecutor

    run_id = int(time.time())
    pool = [f"conc-{run_id}-{k}" for k in range(concurrency * 4)]

    with httpx.Client(
        base_url=base,
        timeout=30.0,
        limits=httpx.Limits(max_connections=concurrency * 2),
    ) as client:
        for k, uid in enumerate(pool):  # warm the cache
            client.post("/recommend_rl", json={"user_id": uid, "signals": _signals(k)})

        latencies: List[float] = []

        def one(i: int) -> float:
            t0 = time.perf_counter()
            client.post(
                "/recommend_rl",
                json={"user_id": pool[i % len(pool)], "signals": _signals(i)},
            ).raise_for_status()
            return (time.perf_counter() - t0) * 1000.0

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool_exec:
            latencies = list(pool_exec.map(one, range(n)))
        elapsed = time.perf_counter() - t_start

    stats = percentiles(latencies)
    stats["concurrency"] = concurrency
    stats["achieved_qps"] = n / elapsed
    return {f"POST /recommend_rl ({concurrency} concurrent)": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="API latency benchmark")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--skip-compute", action="store_true")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    env = dict(os.environ)
    env["REDIS_HOST"] = env.get("REDIS_HOST", "localhost")
    env["REDIS_PORT"] = str(args.redis_port)
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)

    base = f"http://127.0.0.1:{args.port}"

    print("=" * 88)
    print("  API Latency Benchmark — real HTTP")
    print("=" * 88)
    print(f"\n  Requests/arm: {args.n}   warmup: {args.warmup}   port: {args.port}")
    print(f"  Redis: {env['REDIS_HOST']}:{env['REDIS_PORT']}\n")

    print("  starting uvicorn ...", end="", flush=True)
    proc = start_server(args.port, env)
    try:
        if not wait_for_health(base):
            proc.terminate()
            raise SystemExit("server did not become healthy in time")
        print(" ready")

        health = httpx.get(f"{base}/health", timeout=5.0).json()
        cache_info = health.get("feature_cache", {})
        print(
            f"  policy: {health.get('policy')}\n"
            f"  cache : enabled={cache_info.get('cache_enabled')}\n"
        )

        results = run_http_arms(base, args.n, args.warmup)
        results.update(run_concurrent_arm(base, args.n, args.concurrency))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()

    if not args.skip_compute:
        results.update(run_compute_arm(args.n, args.warmup, args.redis_port))

    print()
    print(f"  {'Arm':<40}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}{'QPS':>9}")
    print("  " + "-" * 84)
    for name, r in results.items():
        qps = r.get("achieved_qps", r["qps_serial"])
        print(
            f"  {name:<40}{r['p50_ms']:>8.2f}{r['p95_ms']:>9.2f}"
            f"{r['p99_ms']:>9.2f}{r['mean_ms']:>9.2f}{qps:>9.0f}"
        )

    cold = results.get("POST /recommend_rl (cache miss)")
    warm = results.get("POST /recommend_rl (cache hit)")
    if cold and warm:
        print()
        print(
            f"  Cache speedup: {cold['p50_ms'] / warm['p50_ms']:.1f}x at p50, "
            f"{cold['p99_ms'] / warm['p99_ms']:.1f}x at p99  "
            f"(verified hit: {warm.get('verified_cache_hit')})"
        )

    compute = results.get("compute only (no HTTP, cache hit)")
    if warm and compute:
        print(
            f"  Transport overhead: {warm['p50_ms'] - compute['p50_ms']:+.2f} ms at p50 "
            f"({compute['p50_ms']:.2f} ms compute vs {warm['p50_ms']:.2f} ms end-to-end)"
        )

    conc = next((v for k, v in results.items() if "concurrent" in k), None)
    if conc:
        print(
            f"  Throughput: {conc['achieved_qps']:.0f} req/s at concurrency "
            f"{conc['concurrency']} (p99 {conc['p99_ms']:.2f} ms), single uvicorn worker"
        )

    payload = {
        "config": {
            "n": args.n,
            "warmup": args.warmup,
            "redis_port": args.redis_port,
            "transport": "real HTTP (uvicorn, loopback socket)",
        },
        "results": results,
    }
    out = Path(__file__).parent / "latency_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  Raw results -> {out}")


if __name__ == "__main__":
    main()
