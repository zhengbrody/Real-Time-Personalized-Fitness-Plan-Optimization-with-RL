"""
ProFit AI — A/B Test Runner
===========================

Runs four comparisons over the simulated population and reports significance,
p-values, confidence intervals, effect sizes and power for each:

  1. NeuralLinear vs Random          — is the policy doing anything at all
  2. NeuralLinear vs Rule-based      — the headline claim
  3. NeuralLinear vs Beta TS         — is the learned representation worth it
  4. Linear TS   vs Beta TS          — is conditioning on context worth it

The last two are the ones that can come out negative on a short horizon, and
that is the point of running them: contextual policies pay an exploration cost
up front and only overtake the context-free bandit once they have enough data
to personalise. Running this at 2,000 episodes and at 20,000 gives different
signs on comparison 4 — which is a fact about the horizon, not a bug, and is
worth seeing rather than tuning away.

Usage:
    python scripts/run_ab_test.py [--episodes 5000] [--users 100] [--seed 42]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ab_testing.experiment_framework import ExperimentTracker  # noqa: E402
from src.evaluation.policies import build_policy  # noqa: E402
from src.evaluation.runner import run_policy  # noqa: E402
from src.feature_store.transform import FEATURE_DIM  # noqa: E402
from src.recommendation.action_space import ActionSpace  # noqa: E402

# Arms to compare, as (experiment name, control policy, treatment policy).
COMPARISONS = [
    ("neural_linear_vs_random", "random", "neural_linear"),
    ("neural_linear_vs_rule", "rule_based", "neural_linear"),
    ("neural_linear_vs_beta_ts", "beta_ts", "neural_linear"),
    ("lin_ts_vs_beta_ts", "beta_ts", "lin_ts"),
]

LABELS = {
    "random": "Random Baseline",
    "rule_based": "Rule-based Heuristic",
    "beta_ts": "Beta TS (no context)",
    "lin_ts": "Linear TS",
    "neural_linear": "NeuralLinear",
}


def run_ab_simulation(
    tracker: ExperimentTracker,
    n_episodes: int,
    seed: int,
    users: int = 100,
) -> None:
    """
    Run each policy against the same simulated population and feed the reward
    streams into the A/B framework.

    Every policy is run with the same seed, so they face the same user
    population and the same physiological noise draws.  The comparison is
    unpaired — each policy's actions change the state it subsequently sees, so
    there is no shared per-episode trajectory to pair on — which is exactly the
    setting Welch's t-test is for, and why the framework uses it rather than a
    paired test.
    """
    space = ActionSpace()
    rewards_by_policy = {}

    for key in LABELS:
        policy = build_policy(key, space.get_action_count(), FEATURE_DIM, seed=seed)
        print(f"  Running {LABELS[key]} ...", end="", flush=True)
        records = run_policy(
            policy,
            n_episodes=n_episodes,
            n_users=users,
            seed=seed,
            action_space=space,
            record_scores=False,
        )
        rewards_by_policy[key] = [r.reward for r in records]
        print(" done")

    for name, control, treatment in COMPARISONS:
        exp = tracker.create_experiment(
            name=name,
            control=LABELS[control],
            treatment=LABELS[treatment],
            min_sample_size=100,
            significance_level=0.05,
        )
        for reward in rewards_by_policy[control]:
            exp.add_observation("control", reward)
        for reward in rewards_by_policy[treatment]:
            exp.add_observation("treatment", reward)


def format_report(results: dict) -> str:
    """Format a single experiment's results into a readable report."""
    lines = []
    name = results["experiment_name"]
    ctrl = results["control_strategy"]
    treat = results["treatment_strategy"]

    lines.append(f"  Experiment: {name}")
    lines.append(f"  Control:    {ctrl}")
    lines.append(f"  Treatment:  {treat}")
    lines.append("")

    n_ctrl = results["sample_sizes"]["control"]
    n_treat = results["sample_sizes"]["treatment"]
    lines.append(f"  Sample sizes:    control = {n_ctrl},  treatment = {n_treat}")

    if results["control_mean"] is None:
        lines.append("  [Not enough data to compute statistics]")
        return "\n".join(lines)

    lines.append(f"  Control mean:    {results['control_mean']:.4f}")
    lines.append(f"  Treatment mean:  {results['treatment_mean']:.4f}")
    lines.append(f"  Lift:            {results['lift_pct']:+.2f}%")
    lines.append("")

    lines.append(f"  Welch's t-stat:  {results['t_statistic']:.4f}")
    lines.append(f"  p-value:         {results['p_value']:.6f}")

    ci = results["confidence_interval_95"]
    lines.append(f"  95% CI (diff):   [{ci[0]:.4f}, {ci[1]:.4f}]")
    lines.append(f"  Cohen's d:       {results['effect_size_cohens_d']:.4f}")
    lines.append(f"  Power:           {results['power']:.4f}")
    lines.append("")

    if results["significant"]:
        lines.append("  >>> RESULT: STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        if results["p_value"] < 0.05:
            lines.append("  >>> RESULT: p < 0.05, but min sample size not reached")
        else:
            lines.append("  >>> RESULT: NOT statistically significant")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ProFit AI A/B Test Runner")
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes per agent (default: 500)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=100,
        help="Simulated users interleaved across episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  ProFit AI -- A/B Test Report")
    print("=" * 70)
    print(
        f"\n  Episodes: {args.episodes}  |  Users: {args.users}  |  Seed: {args.seed}"
    )
    print("  NOTE: Simulated environment -- no real users\n")

    tracker = ExperimentTracker()
    run_ab_simulation(
        tracker, n_episodes=args.episodes, seed=args.seed, users=args.users
    )

    summary = tracker.get_summary()

    for i, (name, results) in enumerate(summary.items()):
        print("-" * 70)
        print(format_report(results))
        print()

        # Also report required sample size for an MDE of 0.05
        exp = tracker.get_experiment(name)
        req_n = exp.get_required_sample_size(mde=0.05, power=0.8)
        print(f"  Required n/group for MDE=0.05, power=0.8: {req_n}")
        print()

    print("-" * 70)
    print("  SUMMARY")
    print("-" * 70)
    for name, results in summary.items():
        sig_str = "YES" if results["significant"] else "NO"
        p = results["p_value"]
        p_str = f"{p:.6f}" if p is not None else "N/A"
        d = results["effect_size_cohens_d"]
        d_str = f"{d:.3f}" if d is not None else "N/A"
        print(f"  {name:<25}  significant={sig_str:<4}  p={p_str}  d={d_str}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
