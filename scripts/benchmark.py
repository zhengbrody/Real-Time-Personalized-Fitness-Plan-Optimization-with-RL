"""
ProFit AI — Benchmark Evaluation (single seed)
==============================================

Compares five strategies on the simulated user population:

  1. Random          — uniform over the safety-gated action set
  2. Rule-based      — the hand-crafted heuristic the RL policy replaces
  3. Beta TS         — context-free Thompson Sampling (the ablation control)
  4. Linear TS       — Thompson Sampling on the raw feature vector
  5. NeuralLinear    — Thompson Sampling on a learned representation (PyTorch)

Rows 3-5 share the same posterior machinery and the same safety gate; they
differ only in what they condition on.  So the gap between row 3 and row 4 is
the value of using context at all, and the gap between row 4 and row 5 is the
value of learning a representation rather than assuming linearity.

All results come from simulation with fixed seeds — no real users are involved.
See ``src/simulation/fitness_env.py`` for what the simulator does and does not
model.

Usage
-----
    python scripts/benchmark.py --episodes 5000 --users 100 --seed 42

Output
------
    scripts/benchmark_results.json   raw metrics
    docs/benchmark_single_seed.png   reward + regret curves (unless --no-plot)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.policies import build_policy  # noqa: E402
from src.evaluation.runner import (  # noqa: E402
    compute_metrics,
    convergence_episode,
    pct_improvement,
    run_policy,
)
from src.feature_store.transform import FEATURE_DIM  # noqa: E402
from src.recommendation.action_space import ActionSpace  # noqa: E402

POLICY_KEYS = ["random", "rule_based", "beta_ts", "lin_ts", "neural_linear"]

POLICY_LABELS = {
    "random": "Random",
    "rule_based": "Rule-based",
    "beta_ts": "Beta TS (no context)",
    "lin_ts": "Linear TS",
    "neural_linear": "NeuralLinear",
}

POLICY_COLORS = {
    "random": "#e74c3c",
    "rule_based": "#f39c12",
    "beta_ts": "#9b59b6",
    "lin_ts": "#3498db",
    "neural_linear": "#27ae60",
}


def run_all(
    n_episodes: int, n_users: int, seed: int, verbose: bool = True
) -> Dict[str, dict]:
    """Run every policy and return ``{policy_key: metrics}``."""
    space = ActionSpace()
    results: Dict[str, dict] = {}

    for key in POLICY_KEYS:
        policy = build_policy(key, space.get_action_count(), FEATURE_DIM, seed=seed)
        if verbose:
            print(f"  {POLICY_LABELS[key]:<22}", end="", flush=True)
        t0 = time.time()
        records = run_policy(
            policy,
            n_episodes=n_episodes,
            n_users=n_users,
            seed=seed,
            action_space=space,
        )
        metrics = compute_metrics(records)
        metrics["convergence_episode"] = convergence_episode(records)
        results[key] = metrics
        if verbose:
            print(
                f"reward {metrics['mean_reward']:.4f}  "
                f"regret {metrics['cumulative_regret']:7.1f}  "
                f"({time.time() - t0:.1f}s)"
            )

    return results


def _print_table(results: Dict[str, dict]) -> None:
    rule = results["rule_based"]

    print()
    print("=" * 96)
    print("  RESULTS")
    print("=" * 96)
    header = (
        f"  {'Policy':<22}{'Reward':>9}{'CumRegret':>11}{'FinalRegret':>13}"
        f"{'Optimal%':>10}{'AUC':>8}{'Actions':>9}{'Overtrain%':>12}"
    )
    print(header)
    print("  " + "-" * 92)
    for key in POLICY_KEYS:
        m = results[key]
        print(
            f"  {POLICY_LABELS[key]:<22}"
            f"{m['mean_reward']:>9.4f}"
            f"{m['cumulative_regret']:>11.1f}"
            f"{m['final_regret_rate']:>13.4f}"
            f"{m['optimal_action_rate'] * 100:>9.1f}%"
            f"{m['mean_auc']:>8.3f}"
            f"{m['actions_used']:>6d}/18"
            f"{m['overtraining_rate'] * 100:>11.2f}%"
        )

    print()
    print("  Improvement over the rule-based baseline")
    print("  " + "-" * 92)
    for key in ["beta_ts", "lin_ts", "neural_linear"]:
        m = results[key]
        d_reward = pct_improvement(m, rule)
        d_cum = (
            (m["cumulative_regret"] - rule["cumulative_regret"])
            / rule["cumulative_regret"]
            * 100
        )
        d_final = (
            (m["final_regret_rate"] - rule["final_regret_rate"])
            / rule["final_regret_rate"]
            * 100
        )
        print(
            f"  {POLICY_LABELS[key]:<22}"
            f"reward {d_reward:+7.2f}%   "
            f"cumulative regret {d_cum:+7.1f}%   "
            f"final regret rate {d_final:+7.1f}%"
        )


def _plot(results: Dict[str, dict], out_path: Path, n_episodes: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for key in POLICY_KEYS:
        ax.plot(
            results[key]["rolling_rewards"],
            label=POLICY_LABELS[key],
            color=POLICY_COLORS[key],
            linewidth=1.6,
        )
    ax.set_xlabel("Episode (interleaved user-days)")
    ax.set_ylabel("Rolling mean reward (window=50)")
    ax.set_title("Learning curves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key in POLICY_KEYS:
        ax.plot(
            results[key]["regret_curve"],
            label=POLICY_LABELS[key],
            color=POLICY_COLORS[key],
            linewidth=1.6,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative regret vs oracle")
    ax.set_title("Cumulative regret (lower is better)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"ProFit AI benchmark — {n_episodes} episodes, simulated population",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="ProFit AI benchmark (single seed)")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("=" * 96)
    print("  ProFit AI — Benchmark Evaluation")
    print("=" * 96)
    print(
        f"\n  Episodes: {args.episodes}   Users: {args.users}   "
        f"Seed: {args.seed}   Features: {FEATURE_DIM}"
    )
    print("  NOTE: simulated population — no real users\n")

    results = run_all(args.episodes, args.users, args.seed)
    _print_table(results)

    out_dir = Path(__file__).parent
    payload = {
        "config": {
            "episodes": args.episodes,
            "users": args.users,
            "seed": args.seed,
            "feature_dim": FEATURE_DIM,
            "note": "Simulated population. See src/simulation/fitness_env.py.",
        },
        "results": {
            key: {
                k: v
                for k, v in m.items()
                # Curves are large; keep the JSON readable.
                if k not in ("rolling_rewards", "regret_curve", "action_distribution")
            }
            for key, m in results.items()
        },
        "action_distribution": {
            key: m["action_distribution"] for key, m in results.items()
        },
    }
    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Raw results  -> {json_path}")

    if not args.no_plot:
        plot_path = out_dir.parent / "docs" / "benchmark_single_seed.png"
        if _plot(results, plot_path, args.episodes):
            print(f"  Plot         -> {plot_path}")
        else:
            print("  Plot         -> skipped (matplotlib not installed)")


if __name__ == "__main__":
    main()
