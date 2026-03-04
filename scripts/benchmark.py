"""
ProFit AI - Benchmark Evaluation Script
========================================

Compares three strategies on a simulated fitness environment:
  1. Random Baseline    - selects actions uniformly at random
  2. Rule-based         - hand-crafted heuristics (the project's fallback)
  3. Thompson Sampling  - this project's RL algorithm

All results are from a controlled simulation with fixed random seeds.
They are clearly NOT from real users. This is the honest way to present
portfolio metrics.

Usage:
    python scripts/benchmark.py

Output:
    - Console: per-strategy summary statistics
    - scripts/benchmark_results.json: raw results for README
    - scripts/benchmark_learning_curve.png (optional, if matplotlib available)
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import only the core ML components (no LLM/Kafka/dotenv dependency)
from src.recommendation.action_space import ActionSpace, Action
from src.recommendation.contextual_bandits import ContextualBandit
from src.recommendation.reward_fn import RewardFunction


# ──────────────────────────────────────────────
# Minimal Rule-based recommender (inline, no import chain)
# ──────────────────────────────────────────────

class _MinimalSafetyFilter:
    """Safety filter reproduced inline to avoid heavy import chain."""

    def filter_actions(self, state: Dict, action_space: ActionSpace) -> List[int]:
        all_ids = list(range(action_space.get_action_count()))
        r = state.get("readiness_score", 50)
        f = state.get("fatigue", 5)

        if r < 30 or f > 8:
            # Critical: REST or RECOVERY only
            return [i for i in all_ids
                    if action_space.get_action(i).workout_type in ("REST", "RECOVERY")]
        if f > 6:
            # Fatigued: max LOW intensity
            return [i for i in all_ids
                    if action_space.get_action(i).intensity in ("NONE", "LOW")]
        return all_ids  # all actions allowed


_SAFETY = _MinimalSafetyFilter()


def _rule_based_action(state: Dict, allowed: List[int], action_space: ActionSpace) -> int:
    r = state.get("readiness_score", 50)
    f = state.get("fatigue", 5)
    days = state.get("days_since_training", 1)
    sleep_h = state.get("sleep_duration_hours", 7)

    if r < 40 or sleep_h < 5:
        return 0  # REST

    if f >= 7:
        rec = [a for a in allowed if action_space.get_action(a).workout_type == "RECOVERY"]
        if rec:
            return rec[0]

    if days >= 3:
        med = [a for a in allowed if action_space.get_action(a).intensity == "MEDIUM"]
        if med:
            return med[0]

    low = [a for a in allowed if action_space.get_action(a).intensity == "LOW"]
    if low:
        return low[0]

    return allowed[0] if allowed else 0


# ──────────────────────────────────────────────
# Simulated User Environment
# ──────────────────────────────────────────────

def generate_body_state(rng: np.random.Generator, day: int) -> Dict:
    """
    Simulate realistic body state for a training day.

    Uses a simple periodic model: readiness recovers after rest,
    declines after hard training.  No real users involved.
    """
    week_phase = (day % 7) / 7.0
    base_readiness = 60 + 20 * np.sin(2 * np.pi * week_phase + np.pi)

    readiness = float(np.clip(rng.normal(base_readiness, 12), 20, 100))
    sleep_score = float(np.clip(rng.normal(75, 10), 40, 100))
    hrv = float(np.clip(rng.normal(50 + (readiness - 60) * 0.3, 8), 20, 100))
    resting_hr = float(np.clip(rng.normal(62 - (readiness - 60) * 0.1, 5), 45, 90))
    fatigue = float(np.clip(rng.normal(10 - readiness / 12, 1.5), 1, 10))
    activity_score = float(np.clip(rng.normal(65, 15), 20, 100))
    sleep_hours = float(np.clip(rng.normal(7.2, 0.8), 4, 10))
    days_since = int(rng.integers(0, 3))

    return dict(
        readiness_score=round(readiness),
        sleep_score=round(sleep_score),
        hrv=round(hrv),
        resting_hr=round(resting_hr),
        fatigue=round(fatigue, 1),
        activity_score=round(activity_score),
        sleep_duration_hours=round(sleep_hours, 1),
        days_since_training=days_since,
    )


def compute_optimal_action(state: Dict, action_space: ActionSpace) -> int:
    """
    Oracle: action that would yield highest expected reward for this state.

    Ground-truth user preference in simulation:
      readiness ≥ 70 & fatigue ≤ 4  → HIGH strength (action 8)
      readiness 55-70 & fatigue ≤ 6 → MEDIUM cardio (action 13)
      readiness 40-55               → LOW strength (action 4)
      readiness 30-40               → RECOVERY (action 1)
      readiness < 30                → REST (action 0)
    """
    r = state["readiness_score"]
    f = state["fatigue"]
    if r >= 70 and f <= 4:
        return 8
    elif r >= 55 and f <= 6:
        return 13
    elif r >= 40:
        return 4
    elif r >= 30:
        return 1
    else:
        return 0


def simulate_reward(state: Dict, action: Action, rng: np.random.Generator) -> float:
    """
    Reward a user would receive.

    A mismatch (e.g. HIGH intensity when exhausted) gives negative reward.
    """
    r = state["readiness_score"]
    f = state["fatigue"]
    intensity_score = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    intensity_level = intensity_score.get(action.intensity, 0)

    if r >= 70 and f <= 4:
        ideal = 3
    elif r >= 55 and f <= 6:
        ideal = 2
    elif r >= 40:
        ideal = 1
    else:
        ideal = 0

    mismatch = abs(intensity_level - ideal)
    base = 0.8 - 0.25 * mismatch

    overtraining = intensity_level == 3 and (f > 7 or r < 40)
    if overtraining:
        base -= 0.5

    return float(np.clip(rng.normal(base, 0.1), -1.0, 1.0))


# ──────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────

class RandomAgent:
    name = "Random Baseline"

    def __init__(self, action_space: ActionSpace, rng: np.random.Generator):
        self.action_space = action_space
        self.rng = rng

    def select_action(self, state: Dict) -> int:
        allowed = _SAFETY.filter_actions(state, self.action_space)
        return int(self.rng.choice(allowed))

    def update(self, action_id, state, reward):
        pass


class RuleAgent:
    name = "Rule-based Heuristic"

    def __init__(self, action_space: ActionSpace):
        self.action_space = action_space

    def select_action(self, state: Dict) -> int:
        allowed = _SAFETY.filter_actions(state, self.action_space)
        return _rule_based_action(state, allowed, self.action_space)

    def update(self, action_id, state, reward):
        pass


class ThompsonAgent:
    name = "Thompson Sampling (ProFit AI)"

    def __init__(self, action_space: ActionSpace):
        self.action_space = action_space
        self.bandit = ContextualBandit(action_space)

    def select_action(self, state: Dict) -> int:
        allowed = _SAFETY.filter_actions(state, self.action_space)
        context = np.array([
            state.get("readiness_score", 50) / 100.0,
            state.get("sleep_score", 50) / 100.0,
            state.get("activity_score", 50) / 100.0,
            state.get("hrv", 50) / 100.0,
            state.get("resting_hr", 60) / 100.0,
            state.get("fatigue", 5) / 10.0,
            state.get("days_since_training", 1) / 7.0,
        ])
        return self.bandit.select_action(context, allowed)

    def update(self, action_id, state, reward):
        self.bandit.update(action_id, reward)


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode: int
    action_id: int
    optimal_action_id: int
    reward: float
    is_optimal: bool
    is_overtraining: bool


def run_experiment(agent, n_episodes: int, seed: int, action_space: ActionSpace) -> List[EpisodeResult]:
    rng = np.random.default_rng(seed)
    results = []
    for ep in range(n_episodes):
        state = generate_body_state(rng, ep)
        action_id = agent.select_action(state)
        action = action_space.get_action(action_id)
        reward = simulate_reward(state, action, rng)
        optimal = compute_optimal_action(state, action_space)
        overtraining = (
            action.intensity == "HIGH"
            and (state["fatigue"] > 7 or state["readiness_score"] < 40)
        )
        agent.update(action_id, state, reward)
        results.append(EpisodeResult(
            episode=ep, action_id=action_id, optimal_action_id=optimal,
            reward=reward, is_optimal=(action_id == optimal),
            is_overtraining=overtraining,
        ))
    return results


def rolling_mean(values: List[float], window: int = 50) -> List[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start:i+1])))
    return out


def compute_metrics(results: List[EpisodeResult]) -> Dict:
    rewards = [r.reward for r in results]
    mid = len(rewards) // 2
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "cumulative_reward": float(np.sum(rewards)),
        "optimal_action_rate": float(np.mean([r.is_optimal for r in results])),
        "overtraining_rate": float(np.mean([r.is_overtraining for r in results])),
        "first_half_mean_reward": float(np.mean(rewards[:mid])),
        "second_half_mean_reward": float(np.mean(rewards[mid:])),
        "learning_improvement": float(np.mean(rewards[mid:]) - np.mean(rewards[:mid])),
        "rolling_rewards": rolling_mean(rewards, window=50),
    }


def pct_improvement(a_metrics, b_metrics):
    b = b_metrics["mean_reward"]
    a = a_metrics["mean_reward"]
    return (a - b) / max(abs(b), 1e-9) * 100


def convergence_episode(results: List[EpisodeResult], window=50) -> int:
    rewards = [r.reward for r in results]
    rolling = rolling_mean(rewards, window)
    target = float(np.percentile(rolling[-100:], 25))
    for i, v in enumerate(rolling):
        if v >= target and i > window:
            return i
    return len(results)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ProFit AI Benchmark")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    N, SEED = args.episodes, args.seed
    action_space = ActionSpace()

    print("=" * 70)
    print("  ProFit AI — Benchmark Evaluation")
    print("=" * 70)
    print(f"\n  Episodes : {N}  |  Seed: {SEED}")
    print("  NOTE: Simulated environment — no real users\n")

    rng_random = np.random.default_rng(SEED)
    agents = [
        RandomAgent(action_space, rng_random),
        RuleAgent(action_space),
        ThompsonAgent(action_space),
    ]

    all_metrics = {}
    all_results = {}
    for agent in agents:
        print(f"  Running {agent.name} ...", end="", flush=True)
        t0 = time.time()
        results = run_experiment(agent, N, seed=SEED + 1, action_space=action_space)
        metrics = compute_metrics(results)
        all_results[agent.name] = results
        all_metrics[agent.name] = metrics
        print(f" done ({time.time()-t0:.1f}s)")

    rand_m = all_metrics["Random Baseline"]
    rule_m = all_metrics["Rule-based Heuristic"]
    ts_m   = all_metrics["Thompson Sampling (ProFit AI)"]

    ts_vs_rand = pct_improvement(ts_m, rand_m)
    ts_vs_rule = pct_improvement(ts_m, rule_m)
    conv = convergence_episode(all_results["Thompson Sampling (ProFit AI)"])

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n  {'Metric':<42} {'Random':>10} {'Rule':>10} {'TS (ours)':>12}")
    print("  " + "-" * 68)

    rows = [
        ("Mean Reward",              rand_m["mean_reward"],          rule_m["mean_reward"],          ts_m["mean_reward"],          ".4f"),
        ("Std Reward",               rand_m["std_reward"],           rule_m["std_reward"],           ts_m["std_reward"],           ".4f"),
        ("Cumulative Reward",        rand_m["cumulative_reward"],    rule_m["cumulative_reward"],    ts_m["cumulative_reward"],    ".1f"),
        ("Optimal Action Rate",      rand_m["optimal_action_rate"],  rule_m["optimal_action_rate"],  ts_m["optimal_action_rate"],  ".4f"),
        ("Overtraining Rate",        rand_m["overtraining_rate"],    rule_m["overtraining_rate"],    ts_m["overtraining_rate"],    ".4f"),
        ("Early Mean (ep 0-499)",    rand_m["first_half_mean_reward"],  rule_m["first_half_mean_reward"],  ts_m["first_half_mean_reward"],  ".4f"),
        ("Late Mean  (ep 500-999)",  rand_m["second_half_mean_reward"], rule_m["second_half_mean_reward"], ts_m["second_half_mean_reward"], ".4f"),
    ]

    for label, rv, rulev, tsv, fmt in rows:
        print(f"  {label:<42} {rv:>{10}{fmt}} {rulev:>{10}{fmt}} {tsv:>{12}{fmt}}")

    print()
    print("  KEY FINDINGS")
    print("  " + "-" * 68)
    print(f"  Thompson Sampling vs Random:   {ts_vs_rand:+.1f}% mean reward")
    print(f"  Thompson Sampling vs Rules:    {ts_vs_rule:+.1f}% mean reward")
    print(f"  TS within-run improvement:    {ts_m['learning_improvement']:+.4f} reward (early → late)")
    print(f"  Overtraining rate: TS {ts_m['overtraining_rate']:.1%}  vs  Rule {rule_m['overtraining_rate']:.1%}  vs  Random {rand_m['overtraining_rate']:.1%}")
    print(f"  Convergence approx episode:   ~{conv}")

    # Save results
    output = {
        "config": {
            "n_episodes": N,
            "seed": SEED,
            "note": (
                "Simulated environment. Synthetic body-state data generated with "
                "weekly periodicity + Gaussian noise. Reward encodes domain knowledge: "
                "high-readiness states reward high-intensity; fatigue/low-readiness reward rest."
            ),
        },
        "metrics": {
            name: {k: v for k, v in m.items() if k != "rolling_rewards"}
            for name, m in all_metrics.items()
        },
        "summary": {
            "ts_improvement_vs_random_pct": round(ts_vs_rand, 2),
            "ts_improvement_vs_rule_pct": round(ts_vs_rule, 2),
            "ts_within_run_learning": round(ts_m["learning_improvement"], 4),
            "convergence_episode": conv,
            "ts_overtraining_rate": round(ts_m["overtraining_rate"], 4),
            "rule_overtraining_rate": round(rule_m["overtraining_rate"], 4),
        },
    }

    out_path = ROOT / "scripts" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # Optional plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            colors = {
                "Random Baseline": "#e74c3c",
                "Rule-based Heuristic": "#f39c12",
                "Thompson Sampling (ProFit AI)": "#27ae60",
            }
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("ProFit AI — Simulated Benchmark  (synthetic data, no real users)", fontsize=12)

            for name, metrics in all_metrics.items():
                ax1.plot(metrics["rolling_rewards"], label=name,
                         color=colors[name], linewidth=1.8)
            ax1.axvline(x=500, color="gray", linestyle="--", alpha=0.5, label="Midpoint")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Rolling Mean Reward (window=50)")
            ax1.set_title("Learning Curves")
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            short = ["Random", "Rule-based", "Thompson\nSampling"]
            means = [all_metrics[n]["mean_reward"] for n in [a.name for a in agents]]
            stds  = [all_metrics[n]["std_reward"]  for n in [a.name for a in agents]]
            bars  = ax2.bar(short, means, yerr=stds, capsize=5,
                            color=[colors[a.name] for a in agents],
                            edgecolor="black", linewidth=0.8)
            ax2.set_ylabel("Mean Reward ± Std")
            ax2.set_title("Final Performance")
            ax2.grid(True, axis="y", alpha=0.3)
            for bar, v in zip(bars, means):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=9)

            plt.tight_layout()
            plot_path = ROOT / "scripts" / "benchmark_learning_curve.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"  Plot saved    → {plot_path}")

        except ImportError:
            print("  (matplotlib not available — skipping plot)")

    print()
    print("=" * 70)
    print("  INTERPRETATION FOR README")
    print("=" * 70)
    print(f"""
  Honest claims you can make in portfolio/README:

  Simulation environment (N={N}, seed={SEED}, synthetic data):

    • Thompson Sampling achieves {ts_vs_rand:+.1f}% higher mean reward than
      random selection ({ts_m['mean_reward']:.3f} vs {rand_m['mean_reward']:.3f}).

    • Compared to fixed rule-based heuristics, TS achieves
      {ts_vs_rule:+.1f}% reward difference, {'demonstrating online adaptation' if ts_vs_rule > 0 else 'showing rules are competitive'}.

    • Within-run learning: TS reward improves {ts_m['learning_improvement']:+.4f}
      from first 500 to last 500 episodes (convergence ~ep {conv}).

    • Overtraining rate: TS {ts_m['overtraining_rate']:.1%}, vs Rule {rule_m['overtraining_rate']:.1%},
      vs Random {rand_m['overtraining_rate']:.1%}.

  DO NOT claim 0.85 AUC or "15% improvement" without a source.
  These simulation numbers are traceable and reproducible.
""")


if __name__ == "__main__":
    main()
