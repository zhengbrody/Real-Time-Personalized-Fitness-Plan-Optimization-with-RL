"""
ProFit AI - Convergence Analysis
=================================

Rigorous convergence detection for Thompson Sampling using three independent
methods.  Produces a detailed plot and summary statistics.

Methods:
  1. Rolling mean threshold  (baseline, matches benchmark.py approach)
  2. Change-point detection  (sliding-window t-test for mean stationarity)
  3. Plateau detection       (rolling mean slope stabilisation)

Usage:
    python scripts/convergence_analysis.py              # run fresh benchmark
    python scripts/convergence_analysis.py --from-file  # load cached results

Output:
    - docs/convergence_detail.png   (3-panel diagnostic plot)
    - Console summary table
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ───────────────────────────────────────────────────────
# Convergence detection methods  (pure numpy/scipy)
# ───────────────────────────────────────────────────────


def rolling_mean(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Causal rolling mean (same as benchmark.py)."""
    out = np.empty(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = np.mean(values[start : i + 1])
    return out


def rolling_std(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Causal rolling standard deviation."""
    out = np.empty(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start : i + 1]
        out[i] = np.std(segment) if len(segment) > 1 else 0.0
    return out


def detect_convergence_rolling_mean(
    rewards: np.ndarray, window: int = 50, tail_episodes: int = 100
) -> int:
    """
    Method 1 -- Rolling mean threshold (matches benchmark.py).

    Convergence = first episode (after the warm-up window) where the rolling
    mean reaches the 25th percentile of the last *tail_episodes* rolling
    values.  This is the existing heuristic from benchmark.py kept as a
    baseline.

    Returns the convergence episode, or len(rewards) if never reached.
    """
    rm = rolling_mean(rewards, window)
    tail = rm[-tail_episodes:]
    target = float(np.percentile(tail, 25))
    for i, v in enumerate(rm):
        if v >= target and i > window:
            return i
    return len(rewards)


def detect_convergence_cusum(
    rewards: np.ndarray,
    window: int = 50,
    segment_size: int = 100,
    alpha: float = 0.05,
) -> int:
    """
    Method 2 -- Change-point detection via sliding-window t-test.

    Scans the reward series with a sliding window, comparing consecutive
    non-overlapping segments.  The convergence episode is where adjacent
    segments stop being significantly different (the agent has settled).

    Algorithm:
      1. Divide the episode range into non-overlapping segments of
         *segment_size* episodes.
      2. For each pair of consecutive segments, perform a Welch t-test.
      3. The convergence episode is the *start* of the first segment
         (after the warm-up window) where the t-test p-value exceeds
         *alpha* -- meaning the two segments are NOT significantly
         different (the reward distribution has stabilised).

    Returns the estimated convergence episode, or len(rewards) if the
    agent never stabilises.
    """
    from scipy import stats

    n = len(rewards)
    if n < window + 2 * segment_size:
        return 0

    # Use raw rewards (not rolling mean) for the t-test to preserve
    # the true variance structure
    start = window  # skip warm-up
    for i in range(start, n - 2 * segment_size + 1, segment_size):
        seg_a = rewards[i : i + segment_size]
        seg_b = rewards[i + segment_size : i + 2 * segment_size]
        _, p_value = stats.ttest_ind(seg_a, seg_b, equal_var=False)
        if p_value > alpha:
            return i
    return n


def detect_convergence_plateau(
    rewards: np.ndarray,
    window: int = 50,
    slope_window: int = 100,
    run_length: int = 50,
) -> int:
    """
    Method 3 -- Plateau detection (rolling mean slope stabilisation).

    The idea: once the agent has converged, the rolling mean stops trending
    upward -- its local slope is approximately zero.

    Algorithm:
      1. Compute the rolling mean of rewards.
      2. For each episode i, fit a simple linear slope over the preceding
         *slope_window* points of the rolling mean.
      3. Convergence = first episode (after warm-up) where the absolute
         slope stays below a threshold for *run_length* consecutive episodes.
      4. The threshold is the median absolute slope over the second half of
         the series (i.e., the "settled" regime sets the bar).

    Returns the convergence episode, or len(rewards) if no plateau found.
    """
    rm = rolling_mean(rewards, window)
    n = len(rm)
    if n < window + slope_window + run_length:
        return n

    # Compute local slope at each point using a least-squares fit over slope_window
    slopes = np.zeros(n)
    x = np.arange(slope_window, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    for i in range(slope_window, n):
        y = rm[i - slope_window : i]
        slopes[i] = np.sum((x - x_mean) * (y - y.mean())) / x_var

    # Threshold: median absolute slope in the second half (assumed converged region)
    start_idx = max(slope_window, n // 2)
    abs_slopes_tail = np.abs(slopes[start_idx:])
    threshold = float(np.median(abs_slopes_tail)) + float(np.std(abs_slopes_tail))

    # Handle edge case: if threshold is essentially zero, use a small fraction
    # of the overall range divided by the slope window
    rm_range = float(np.max(rm[window:]) - np.min(rm[window:]))
    min_threshold = rm_range / (slope_window * 10) if rm_range > 0 else 1e-6
    threshold = max(threshold, min_threshold)

    consecutive = 0
    for i in range(slope_window, n):
        if abs(slopes[i]) <= threshold:
            consecutive += 1
            if consecutive >= run_length:
                return i - run_length + 1
        else:
            consecutive = 0
    return n


# ───────────────────────────────────────────────────────
# Running / loading benchmark data
# ───────────────────────────────────────────────────────


def _run_fresh_benchmark(
    n_episodes: int = 5000, seed: int = 42, users: int = 100
) -> Dict[str, np.ndarray]:
    """Run a fresh benchmark and return per-policy reward arrays."""
    from scripts.benchmark import POLICY_LABELS
    from src.evaluation.policies import build_policy
    from src.evaluation.runner import run_policy
    from src.feature_store.transform import FEATURE_DIM
    from src.recommendation.action_space import ActionSpace

    # The Beta-Bernoulli bandit samples from the legacy global RNG, so it has
    # to be seeded separately for the run to be reproducible.
    np.random.seed(seed)

    space = ActionSpace()
    results: Dict[str, np.ndarray] = {}
    for key in ("random", "rule_based", "beta_ts", "lin_ts", "neural_linear"):
        policy = build_policy(key, space.get_action_count(), FEATURE_DIM, seed=seed)
        print(f"  Running {POLICY_LABELS[key]} ...", end="", flush=True)
        records = run_policy(
            policy,
            n_episodes=n_episodes,
            n_users=users,
            seed=seed,
            action_space=space,
            record_scores=False,
        )
        results[POLICY_LABELS[key]] = np.array([r.reward for r in records])
        print(" done")
    return results


def _load_cached_results(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load per-episode rewards from a cached multi-seed run, if available.

    The aggregated results file stores scalar summaries per seed, not the
    per-episode reward series these detectors need — carrying 10 seeds × 5
    policies × 5,000 rewards would bloat it by two orders of magnitude for data
    only this script consumes.  So the cache is used when it happens to carry
    curves and is otherwise reported as unusable, rather than being silently
    reinterpreted into something with the right shape and the wrong meaning.
    """
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    # Legacy format: {"seeds": {seed: {policy: [per-episode rewards]}}}
    if "seeds" in data:
        out: Dict[str, np.ndarray] = {}
        for _, strategies in data["seeds"].items():
            for name, rewards in strategies.items():
                arr = np.asarray(rewards, dtype=float)
                out[name] = out.get(name, np.zeros_like(arr)) + arr
        for name in out:
            out[name] /= len(data["seeds"])
        return out or None

    return None


# ───────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────


def generate_plot(
    results: Dict[str, np.ndarray],
    convergence_points: Dict[str, int],
    output_path: Path,
    window: int = 50,
) -> None:
    """Generate the 3-panel convergence diagnostic plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts_key = None
    for k in results:
        if "thompson" in k.lower() or "ts" in k.lower():
            ts_key = k
            break
    if ts_key is None:
        ts_key = list(results.keys())[-1]

    ts_rewards = results[ts_key]
    rm = rolling_mean(ts_rewards, window)
    rstd = rolling_std(ts_rewards, window)
    episodes = np.arange(len(ts_rewards))

    colors = {
        "Random Baseline": "#e74c3c",
        "Rule-based Heuristic": "#f39c12",
        "Thompson Sampling (ProFit AI)": "#27ae60",
    }
    default_color = "#27ae60"

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        "Thompson Sampling Convergence Analysis  (simulated data)",
        fontsize=14,
        fontweight="bold",
    )

    # ── Top panel: raw rewards + rolling mean + convergence markers ──
    ax = axes[0]
    ax.scatter(episodes, ts_rewards, alpha=0.08, s=3, color="gray", label="Raw rewards")
    ax.plot(
        episodes,
        rm,
        color=default_color,
        linewidth=2,
        label=f"Rolling mean (w={window})",
    )

    markers = {
        "Rolling mean threshold": ("v", "#e74c3c"),
        "Change-point (CUSUM)": ("D", "#3498db"),
        "Plateau detection": ("s", "#9b59b6"),
    }
    for method_name, ep in convergence_points.items():
        marker, color = markers.get(method_name, ("o", "black"))
        if 0 < ep < len(rm):
            ax.axvline(x=ep, color=color, linestyle="--", alpha=0.6, linewidth=1)
            ax.plot(
                ep,
                rm[ep],
                marker=marker,
                color=color,
                markersize=10,
                zorder=5,
                label=f"{method_name}: ep {ep}",
            )

    ax.set_ylabel("Reward")
    ax.set_title("Rewards and Rolling Mean with Convergence Points")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Middle panel: rolling std + rolling mean slope ──
    ax = axes[1]
    ax.plot(
        episodes,
        rstd,
        color="#8e44ad",
        linewidth=1.5,
        alpha=0.6,
        label=f"Rolling std (w={window})",
    )

    # Overlay rolling mean slope (used by plateau detection)
    slope_window = 100
    slopes = np.zeros(len(rm))
    x = np.arange(slope_window, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    for i in range(slope_window, len(rm)):
        y = rm[i - slope_window : i]
        slopes[i] = np.sum((x - x_mean) * (y - y.mean())) / x_var

    ax2 = ax.twinx()
    ax2.plot(
        episodes,
        np.abs(slopes),
        color="#2980b9",
        linewidth=1.2,
        alpha=0.7,
        label="|Rolling mean slope|",
    )
    ax2.set_ylabel("|Slope of Rolling Mean|", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")

    plateau_ep = convergence_points.get("Plateau detection", None)
    if plateau_ep and 0 < plateau_ep < len(rstd):
        ax.axvline(x=plateau_ep, color="#9b59b6", linestyle="--", alpha=0.6)
        ax.plot(
            plateau_ep,
            rstd[plateau_ep],
            "s",
            color="#9b59b6",
            markersize=10,
            zorder=5,
            label=f"Plateau start: ep {plateau_ep}",
        )

    ax.set_ylabel("Rolling Std", color="#8e44ad")
    ax.tick_params(axis="y", labelcolor="#8e44ad")
    ax.set_title("Reward Variance and Rolling Mean Slope Stabilisation")
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Bottom panel: cumulative rewards ──
    ax = axes[2]
    for name, rewards in results.items():
        cum = np.cumsum(rewards)
        color = colors.get(name, "gray")
        ax.plot(np.arange(len(cum)), cum, color=color, linewidth=1.8, label=name)

    # Mark divergence point: where Thompson first permanently leads
    if len(results) >= 2:
        other_keys = [k for k in results if k != ts_key]
        if other_keys:
            ts_cum = np.cumsum(results[ts_key])
            best_other_cum = np.cumsum(results[other_keys[0]])
            for k in other_keys[1:]:
                c = np.cumsum(results[k])
                best_other_cum = np.maximum(best_other_cum, c)
            # Find last crossing point
            diff = ts_cum - best_other_cum
            crossings = np.where(np.diff(np.sign(diff)))[0]
            if len(crossings) > 0:
                divergence_ep = int(crossings[-1]) + 1
                ax.axvline(
                    x=divergence_ep,
                    color="black",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                    label=f"TS diverges at ep {divergence_ep}",
                )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Comparison (divergence point)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved -> {output_path}")


# ───────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ProFit AI Convergence Analysis")
    parser.add_argument(
        "--from-file",
        action="store_true",
        help="Load from benchmark_multi_seed_results.json instead of running fresh",
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    # ── Load or generate data ──
    results = None
    if args.from_file:
        cached_path = PROJECT_ROOT / "scripts" / "benchmark_multi_seed_results.json"
        results = _load_cached_results(cached_path)
        if results is not None:
            print(f"  Loaded cached results from {cached_path}")
        else:
            print(
                f"  No per-episode curves in {cached_path.name}; "
                "running a fresh benchmark ..."
            )

    if results is None:
        results = _run_fresh_benchmark(args.episodes, args.seed, args.users)

    # ── Analyse the deployed policy ──
    # NeuralLinear is what the service actually runs, so it is the one whose
    # convergence matters; fall back to any bandit row if it is absent.
    ts_key = next(
        (k for k in results if "neurallinear" in k.lower().replace(" ", "")), None
    )
    if ts_key is None:
        ts_key = next((k for k in results if "ts" in k.lower()), list(results)[-1])

    ts_rewards = results[ts_key]
    window = args.window

    # ── Run all three detection methods ──
    ep_rolling = detect_convergence_rolling_mean(ts_rewards, window=window)
    ep_cusum = detect_convergence_cusum(ts_rewards, window=window)
    ep_plateau = detect_convergence_plateau(ts_rewards, window=window)

    convergence_points = {
        "Rolling mean threshold": ep_rolling,
        "Change-point (CUSUM)": ep_cusum,
        "Plateau detection": ep_plateau,
    }

    episodes_list = [ep_rolling, ep_cusum, ep_plateau]
    # Filter out degenerate values (0 or len) for the mean estimate
    valid = [e for e in episodes_list if 0 < e < len(ts_rewards)]
    if valid:
        mean_est = float(np.mean(valid))
        std_est = float(np.std(valid))
    else:
        mean_est = float(np.mean(episodes_list))
        std_est = float(np.std(episodes_list))

    # ── Print summary ──
    bar = "\u2500" * 45
    print()
    print(f"  Convergence Analysis ({ts_key})")
    print(f"  {bar}")
    print(f"  Rolling mean threshold:    ep {ep_rolling}")
    print(f"  Change-point (CUSUM):      ep {ep_cusum}")
    print(f"  Plateau detection:         ep {ep_plateau}")
    print(f"  Mean convergence estimate: ep {mean_est:.0f} +/- {std_est:.0f}")
    print()

    # ── Generate plot ──
    if not args.no_plot:
        plot_path = PROJECT_ROOT / "docs" / "convergence_detail.png"
        generate_plot(results, convergence_points, plot_path, window=window)

    return convergence_points, mean_est, std_est


if __name__ == "__main__":
    main()
