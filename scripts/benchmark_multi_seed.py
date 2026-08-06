"""
ProFit AI — Multi-Seed Benchmark
================================

Runs the full policy comparison across independent seeds and reports mean ± std
with 95% confidence intervals.

A single seed proves nothing about a stochastic policy: Thompson Sampling's
early exploration is random, and on a short horizon that alone can move mean
reward by several percent.  Every number quoted in the README comes from this
script, across 10 seeds, so the reported effect is separable from seed noise.

Each seed constructs a fresh simulator, so the user population differs between
seeds as well as the exploration draws — the variation being measured is over
*populations and runs*, not just over random draws for one fixed population.

Usage
-----
    python scripts/benchmark_multi_seed.py --num_seeds 10 --episodes 5000

Output
------
    scripts/benchmark_multi_seed_results.json
    docs/learning_curves.png
    docs/regret_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark import (  # noqa: E402
    POLICY_COLORS,
    POLICY_KEYS,
    POLICY_LABELS,
    run_all,
)
from src.feature_store.transform import FEATURE_DIM  # noqa: E402

SCALAR_METRICS = [
    "mean_reward",
    "std_reward",
    "cumulative_reward",
    "cumulative_regret",
    "mean_regret",
    "final_regret_rate",
    "optimal_action_rate",
    "overtraining_rate",
    "mean_auc",
    "actions_used",
    "actions_used_above_1pct",
    "learning_improvement",
    "convergence_episode",
]


def aggregate(per_seed: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    """
    Collapse ``{seed_label: {policy: metrics}}`` into
    ``{policy: {metric: {mean, std, ci_95}}}``.

    The CI is a normal approximation, which is adequate at n=10 for the effect
    sizes involved here; it is reported so the reader can see the spread, not
    to support a formal hypothesis test (that lives in the A/B framework).
    """
    out: Dict[str, dict] = {}
    for policy in POLICY_KEYS:
        out[policy] = {}
        for metric in SCALAR_METRICS:
            values = np.array(
                [per_seed[s][policy][metric] for s in per_seed], dtype=float
            )
            n = len(values)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            out[policy][metric] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "ci_95": [round(mean - half, 6), round(mean + half, 6)],
            }
    return out


def relative_to_baseline(
    per_seed: Dict[str, Dict[str, dict]], baseline: str = "rule_based"
) -> Dict[str, dict]:
    """
    Per-seed relative improvements, aggregated.

    Computing the ratio *within* each seed and then averaging — rather than
    taking the ratio of the averages — keeps the baseline's own seed-to-seed
    variation from leaking into the reported effect.
    """
    out: Dict[str, dict] = {}
    for policy in POLICY_KEYS:
        if policy == baseline:
            continue
        reward_lift, cum_regret, final_regret = [], [], []
        for s in per_seed:
            b = per_seed[s][baseline]
            p = per_seed[s][policy]
            reward_lift.append(
                (p["mean_reward"] - b["mean_reward"]) / abs(b["mean_reward"]) * 100
            )
            cum_regret.append(
                (p["cumulative_regret"] - b["cumulative_regret"])
                / b["cumulative_regret"]
                * 100
            )
            final_regret.append(
                (p["final_regret_rate"] - b["final_regret_rate"])
                / b["final_regret_rate"]
                * 100
            )
        out[policy] = {
            "reward_lift_pct": {
                "mean": round(float(np.mean(reward_lift)), 4),
                "std": round(float(np.std(reward_lift, ddof=1)), 4),
            },
            "cumulative_regret_change_pct": {
                "mean": round(float(np.mean(cum_regret)), 4),
                "std": round(float(np.std(cum_regret, ddof=1)), 4),
            },
            "final_regret_change_pct": {
                "mean": round(float(np.mean(final_regret)), 4),
                "std": round(float(np.std(final_regret, ddof=1)), 4),
            },
        }
    return out


def _plot_bands(
    curves: Dict[str, List[List[float]]],
    ylabel: str,
    title: str,
    out_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    for key in POLICY_KEYS:
        arr = np.array(curves[key], dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(
            x, mean, label=POLICY_LABELS[key], color=POLICY_COLORS[key], linewidth=1.8
        )
        ax.fill_between(x, mean - std, mean + std, color=POLICY_COLORS[key], alpha=0.15)

    ax.set_xlabel("Episode (interleaved user-days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="ProFit AI multi-seed benchmark")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    seeds = list(range(1, args.num_seeds + 1))

    print("=" * 96)
    print("  ProFit AI — Multi-Seed Benchmark")
    print("=" * 96)
    print(
        f"\n  Seeds: {seeds}\n  Episodes/seed: {args.episodes}   "
        f"Users: {args.users}   Features: {FEATURE_DIM}\n"
    )

    per_seed: Dict[str, Dict[str, dict]] = {}
    reward_curves: Dict[str, List[List[float]]] = {k: [] for k in POLICY_KEYS}
    regret_curves: Dict[str, List[List[float]]] = {k: [] for k in POLICY_KEYS}

    t_start = time.time()
    for seed in seeds:
        print(f"  --- seed {seed} ---")
        results = run_all(args.episodes, args.users, seed)
        per_seed[f"seed_{seed}"] = {
            k: {m: v[m] for m in SCALAR_METRICS} for k, v in results.items()
        }
        for k in POLICY_KEYS:
            reward_curves[k].append(results[k]["rolling_rewards"])
            regret_curves[k].append(results[k]["regret_curve"])

    aggregated = aggregate(per_seed)
    relative = relative_to_baseline(per_seed)

    print()
    print("=" * 96)
    print("  AGGREGATED ACROSS SEEDS  (mean ± std)")
    print("=" * 96)
    print(
        f"  {'Policy':<22}{'Reward':>16}{'CumRegret':>16}"
        f"{'FinalRegret':>16}{'AUC':>14}{'Actions':>10}"
    )
    print("  " + "-" * 92)
    for key in POLICY_KEYS:
        a = aggregated[key]
        print(
            f"  {POLICY_LABELS[key]:<22}"
            f"{a['mean_reward']['mean']:>9.4f} ±{a['mean_reward']['std']:<5.4f}"
            f"{a['cumulative_regret']['mean']:>9.1f} ±{a['cumulative_regret']['std']:<5.1f}"
            f"{a['final_regret_rate']['mean']:>9.4f} ±{a['final_regret_rate']['std']:<5.4f}"
            f"{a['mean_auc']['mean']:>8.3f} ±{a['mean_auc']['std']:<4.3f}"
            f"{a['actions_used']['mean']:>7.1f}/18"
        )

    print()
    print("  vs rule-based baseline (per-seed ratios, then averaged)")
    print("  " + "-" * 92)
    for key, r in relative.items():
        print(
            f"  {POLICY_LABELS[key]:<22}"
            f"reward {r['reward_lift_pct']['mean']:+7.2f}% ± {r['reward_lift_pct']['std']:.2f}   "
            f"cum regret {r['cumulative_regret_change_pct']['mean']:+7.2f}% ± "
            f"{r['cumulative_regret_change_pct']['std']:.2f}   "
            f"final regret {r['final_regret_change_pct']['mean']:+7.2f}%"
        )

    payload = {
        "config": {
            "num_seeds": args.num_seeds,
            "seeds": seeds,
            "episodes_per_seed": args.episodes,
            "users": args.users,
            "feature_dim": FEATURE_DIM,
            "note": "Simulated population. See src/simulation/fitness_env.py.",
        },
        "aggregated": aggregated,
        "relative_to_rule_based": relative,
        "per_seed": per_seed,
    }
    out_path = Path(__file__).parent / "benchmark_multi_seed_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Raw results -> {out_path}")

    if not args.no_plot:
        docs = Path(__file__).parent.parent / "docs"
        ok1 = _plot_bands(
            reward_curves,
            "Rolling mean reward (window=50)",
            f"Learning curves — mean ± 1 std across {args.num_seeds} seeds",
            docs / "learning_curves.png",
        )
        ok2 = _plot_bands(
            regret_curves,
            "Cumulative regret vs oracle",
            f"Cumulative regret — mean ± 1 std across {args.num_seeds} seeds",
            docs / "regret_curves.png",
        )
        if ok1 and ok2:
            print(
                f"  Plots       -> {docs}/learning_curves.png, {docs}/regret_curves.png"
            )
        else:
            print("  Plots       -> skipped (matplotlib not installed)")

    print(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
