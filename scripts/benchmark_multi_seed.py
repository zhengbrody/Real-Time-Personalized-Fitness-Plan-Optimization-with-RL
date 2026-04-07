"""
ProFit AI - Multi-Seed Benchmark Evaluation
============================================

Runs the benchmark across multiple random seeds and aggregates results
with mean, std, and 95% confidence intervals.

Reuses all simulation logic from benchmark.py.

Usage:
    python scripts/benchmark_multi_seed.py --num_seeds 10 --episodes 1000

Output:
    - scripts/benchmark_multi_seed_results.json
    - docs/learning_curves.png
    - docs/convergence_analysis.png
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse everything from the existing benchmark
from scripts.benchmark import (  # noqa: E402
    ActionSpace,
    RandomAgent,
    RuleAgent,
    ThompsonAgent,
    run_experiment,
    compute_metrics,
    convergence_episode,
    rolling_mean,
    pct_improvement,
)

# ──────────────────────────────────────────────
# Agent key mapping (short keys for JSON output)
# ──────────────────────────────────────────────

AGENT_KEYS = {
    "Random Baseline": "random",
    "Rule-based Heuristic": "rule_based",
    "Thompson Sampling (ProFit AI)": "thompson",
}

AGENT_COLORS = {
    "random": "#e74c3c",
    "rule_based": "#f39c12",
    "thompson": "#27ae60",
}

AGENT_LABELS = {
    "random": "Random Baseline",
    "rule_based": "Rule-based Heuristic",
    "thompson": "Thompson Sampling",
}


def run_single_seed(seed: int, n_episodes: int):
    """
    Run all three agents for a single seed.
    Returns (metrics_dict, rolling_rewards_dict, convergence_dict).
    """
    action_space = ActionSpace()

    rng_random = np.random.default_rng(seed)
    agents = [
        RandomAgent(action_space, rng_random),
        RuleAgent(action_space),
        ThompsonAgent(action_space),
    ]

    seed_metrics = {}
    seed_rolling = {}
    seed_convergence = {}

    for agent in agents:
        key = AGENT_KEYS[agent.name]
        results = run_experiment(agent, n_episodes, seed=seed + 1, action_space=action_space)
        metrics = compute_metrics(results)
        conv_ep = convergence_episode(results)

        seed_metrics[key] = {
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "cumulative_reward": metrics["cumulative_reward"],
            "optimal_action_rate": metrics["optimal_action_rate"],
            "overtraining_rate": metrics["overtraining_rate"],
            "convergence_episode": conv_ep,
        }
        seed_rolling[key] = metrics["rolling_rewards"]
        seed_convergence[key] = conv_ep

    return seed_metrics, seed_rolling, seed_convergence


def aggregate_across_seeds(per_seed_data: dict, metric_names: list):
    """
    Aggregate a metric across seeds: mean, std, 95% CI.
    per_seed_data: {agent_key: [list of values, one per seed]}
    Returns: {agent_key: {metric: {mean, std, ci_95}}}
    """
    aggregated = {}
    for agent_key in ["random", "rule_based", "thompson"]:
        aggregated[agent_key] = {}
        for metric in metric_names:
            values = [per_seed_data[seed_label][agent_key][metric]
                      for seed_label in per_seed_data]
            arr = np.array(values)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            n = len(arr)
            # 95% CI using t-distribution approximation (for n>=10, z~1.96 is fine)
            ci_half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            aggregated[agent_key][metric] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "ci_95": [round(mean - ci_half, 6), round(mean + ci_half, 6)],
            }
    return aggregated


def generate_learning_curves(all_rolling: dict, all_convergence: dict,
                             n_episodes: int, output_path: Path):
    """
    Generate publication-quality learning curves with confidence bands.
    all_rolling: {seed_label: {agent_key: [rolling_reward_per_episode]}}
    all_convergence: {seed_label: {agent_key: convergence_episode}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))

    episodes = np.arange(n_episodes)

    for agent_key in ["random", "rule_based", "thompson"]:
        # Collect rolling rewards across all seeds
        all_curves = []
        for seed_label in all_rolling:
            curve = all_rolling[seed_label][agent_key]
            all_curves.append(curve)

        all_curves = np.array(all_curves)  # shape: (num_seeds, n_episodes)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)

        color = AGENT_COLORS[agent_key]
        label = AGENT_LABELS[agent_key]

        ax.plot(episodes, mean_curve, label=label, color=color, linewidth=2.0)
        ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                        color=color, alpha=0.15)

    # Mark Thompson convergence point (mean across seeds)
    ts_conv_episodes = [all_convergence[s]["thompson"] for s in all_convergence]
    mean_conv = int(np.mean(ts_conv_episodes))
    ax.axvline(x=mean_conv, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate(f"TS converges ~ep {mean_conv}",
                xy=(mean_conv, ax.get_ylim()[0]),
                xytext=(mean_conv + 40, 0.15),
                fontsize=9, color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.0))

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Rolling Mean Reward (window=50)", fontsize=11)
    ax.set_title("Learning Curves Across Seeds (mean +/- 1 std)", fontsize=12, pad=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_convergence_analysis(all_convergence: dict, output_path: Path):
    """
    Scatter plot of per-seed convergence episodes + mean bar.
    all_convergence: {seed_label: {agent_key: convergence_episode}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts_convs = [all_convergence[s]["thompson"] for s in sorted(all_convergence)]
    seeds = list(range(1, len(ts_convs) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    mean_conv = float(np.mean(ts_convs))
    std_conv = float(np.std(ts_convs, ddof=1)) if len(ts_convs) > 1 else 0.0

    # Scatter for each seed
    ax.scatter(seeds, ts_convs, color="#27ae60", s=70, zorder=5,
               edgecolors="white", linewidths=1.2, label="Per-seed convergence")

    # Mean bar
    ax.axhline(y=mean_conv, color="#27ae60", linewidth=2.0, linestyle="-",
               alpha=0.8, label=f"Mean = {mean_conv:.0f}")
    # Std band
    ax.axhspan(mean_conv - std_conv, mean_conv + std_conv,
               color="#27ae60", alpha=0.1, label=f"+/- 1 std ({std_conv:.0f})")

    ax.set_xlabel("Seed", fontsize=11)
    ax.set_ylabel("Convergence Episode", fontsize=11)
    ax.set_title("Thompson Sampling Convergence Analysis", fontsize=12, pad=10)
    ax.set_xticks(seeds)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary_table(aggregated: dict, summary: dict, num_seeds: int, n_episodes: int):
    """Print a clean summary table to stdout."""
    print()
    print("=" * 74)
    print("  ProFit AI - Multi-Seed Benchmark Results")
    print("=" * 74)
    print(f"\n  Seeds: {num_seeds}  |  Episodes per seed: {n_episodes}")
    print("  NOTE: Simulated environment - no real users\n")

    metrics_to_show = [
        ("Mean Reward", "mean_reward", ".4f"),
        ("Std Reward", "std_reward", ".4f"),
        ("Cumulative Reward", "cumulative_reward", ".1f"),
        ("Optimal Action Rate", "optimal_action_rate", ".4f"),
        ("Overtraining Rate", "overtraining_rate", ".4f"),
        ("Convergence Episode", "convergence_episode", ".0f"),
    ]

    header = f"  {'Metric':<30} {'Random':>14} {'Rule-based':>14} {'Thompson':>14}"
    print(header)
    print("  " + "-" * 72)

    for label, key, fmt in metrics_to_show:
        vals = []
        for agent_key in ["random", "rule_based", "thompson"]:
            m = aggregated[agent_key][key]["mean"]
            s = aggregated[agent_key][key]["std"]
            vals.append(f"{m:{fmt}} +/-{s:{fmt}}")
        print(f"  {label:<30} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    print()
    print("  KEY FINDINGS (across seeds)")
    print("  " + "-" * 72)
    ts_vs_rand = summary["ts_vs_random_pct"]
    ts_vs_rule = summary["ts_vs_rule_pct"]
    print(f"  TS vs Random:  {ts_vs_rand['mean']:+.1f}% +/- {ts_vs_rand['std']:.1f}% mean reward")
    print(f"  TS vs Rule:    {ts_vs_rule['mean']:+.1f}% +/- {ts_vs_rule['std']:.1f}% mean reward")
    print()


def main():
    parser = argparse.ArgumentParser(description="ProFit AI Multi-Seed Benchmark")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds to run (seeds 1..N)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Episodes per seed")
    args = parser.parse_args()

    num_seeds = args.num_seeds
    n_episodes = args.episodes
    seeds = list(range(1, num_seeds + 1))

    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  ProFit AI - Multi-Seed Benchmark")
    print("=" * 74)
    print(f"\n  Running {num_seeds} seeds x {n_episodes} episodes each ...\n")

    # Collect per-seed results
    per_seed = {}          # {seed_label: {agent_key: {metric: value}}}
    all_rolling = {}       # {seed_label: {agent_key: [rolling_rewards]}}
    all_convergence = {}   # {seed_label: {agent_key: conv_episode}}

    for seed in seeds:
        label = f"seed_{seed}"
        t0 = time.time()
        metrics, rolling, convergence = run_single_seed(seed, n_episodes)
        elapsed = time.time() - t0
        per_seed[label] = metrics
        all_rolling[label] = rolling
        all_convergence[label] = convergence
        print(f"    Seed {seed:>2}/{num_seeds} done ({elapsed:.1f}s)")

    # Aggregate across seeds
    metric_names = [
        "mean_reward", "std_reward", "cumulative_reward",
        "optimal_action_rate", "overtraining_rate", "convergence_episode",
    ]
    aggregated = aggregate_across_seeds(per_seed, metric_names)

    # Compute TS improvement percentages per seed, then aggregate
    ts_vs_random_pcts = []
    ts_vs_rule_pcts = []
    for label in per_seed:
        ts_mean = per_seed[label]["thompson"]["mean_reward"]
        rand_mean = per_seed[label]["random"]["mean_reward"]
        rule_mean = per_seed[label]["rule_based"]["mean_reward"]
        ts_vs_random_pcts.append((ts_mean - rand_mean) / max(abs(rand_mean), 1e-9) * 100)
        ts_vs_rule_pcts.append((ts_mean - rule_mean) / max(abs(rule_mean), 1e-9) * 100)

    summary = {
        "ts_vs_random_pct": {
            "mean": round(float(np.mean(ts_vs_random_pcts)), 2),
            "std": round(float(np.std(ts_vs_random_pcts, ddof=1)), 2) if num_seeds > 1 else 0.0,
        },
        "ts_vs_rule_pct": {
            "mean": round(float(np.mean(ts_vs_rule_pcts)), 2),
            "std": round(float(np.std(ts_vs_rule_pcts, ddof=1)), 2) if num_seeds > 1 else 0.0,
        },
    }

    # Print summary table
    print_summary_table(aggregated, summary, num_seeds, n_episodes)

    # Build JSON output
    output = {
        "config": {
            "num_seeds": num_seeds,
            "episodes_per_seed": n_episodes,
            "seeds": seeds,
        },
        "per_seed": {
            label: per_seed[label] for label in sorted(per_seed)
        },
        "aggregated": aggregated,
        "summary": summary,
    }

    json_path = Path(__file__).parent / "benchmark_multi_seed_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved -> {json_path}")

    # Generate plots
    try:
        lc_path = docs_dir / "learning_curves.png"
        generate_learning_curves(all_rolling, all_convergence, n_episodes, lc_path)
        print(f"  Plot saved    -> {lc_path}")

        ca_path = docs_dir / "convergence_analysis.png"
        generate_convergence_analysis(all_convergence, ca_path)
        print(f"  Plot saved    -> {ca_path}")
    except ImportError:
        print("  (matplotlib not available - skipping plots)")

    print()
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
