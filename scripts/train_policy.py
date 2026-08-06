"""
Train and Persist the Serving Policy
====================================

Trains a NeuralLinear bandit against the simulator and writes a checkpoint the
API server loads at startup.

The checkpoint carries the encoder weights *and* the per-action posteriors, not
just the network.  A NeuralLinear policy whose posteriors were dropped and
re-initialised would come up maximally uncertain and explore as if it had never
seen a user — restoring the network alone silently resets the thing that makes
it a bandit.

Usage
-----
    python scripts/train_policy.py --episodes 20000 --users 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.policies import NeuralLinearPolicy  # noqa: E402
from src.evaluation.runner import compute_metrics, run_policy  # noqa: E402
from src.feature_store.transform import FEATURE_DIM, FEATURE_NAMES  # noqa: E402
from src.recommendation.action_space import ActionSpace  # noqa: E402

DEFAULT_PATH = Path(__file__).parent.parent / "models" / "neural_linear.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the serving policy")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    space = ActionSpace()
    policy = NeuralLinearPolicy(space.get_action_count(), FEATURE_DIM, seed=args.seed)

    print(f"Training NeuralLinear: {args.episodes} episodes, {args.users} users")
    t0 = time.time()
    records = run_policy(
        policy,
        n_episodes=args.episodes,
        n_users=args.users,
        seed=args.seed,
        action_space=space,
    )
    metrics = compute_metrics(records)
    elapsed = time.time() - t0

    n = len(records)
    tail = [r.reward for r in records[-n // 5 :]]
    print(
        f"  done in {elapsed:.1f}s  |  mean reward {metrics['mean_reward']:.4f}  "
        f"final-fifth reward {np.mean(tail):.4f}  "
        f"final regret rate {metrics['final_regret_rate']:.4f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = policy.model.state_dict()
    torch.save(
        {
            "encoder": state["encoder"],
            "A": state["A"],
            "b": state["b"],
            "mu": state["mu"],
            "alpha": state["alpha"],
            "beta": state["beta"],
            "counts": state["counts"],
            "feature_dim": FEATURE_DIM,
            "feature_names": FEATURE_NAMES,
            "n_actions": space.get_action_count(),
            "training": {
                "episodes": args.episodes,
                "users": args.users,
                "seed": args.seed,
                "mean_reward": metrics["mean_reward"],
                "final_regret_rate": metrics["final_regret_rate"],
            },
        },
        args.out,
    )
    print(f"  checkpoint -> {args.out}")

    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "feature_dim": FEATURE_DIM,
                "feature_names": FEATURE_NAMES,
                "n_actions": space.get_action_count(),
                "metrics": {
                    k: v
                    for k, v in metrics.items()
                    if k
                    not in ("rolling_rewards", "regret_curve", "action_distribution")
                },
            },
            indent=2,
        )
    )
    print(f"  metadata   -> {meta_path}")


if __name__ == "__main__":
    main()
