"""
Off-Policy Estimator Calibration Study
======================================

Measures how wrong each off-policy estimator actually is.

The point
--------
On real logged data you can compute IPS, SNIPS, DM and DR — and you have no way
of knowing which one to believe, because the true value of the target policy is
exactly the thing you were trying to estimate.  Papers report estimator
comparisons on benchmark datasets by *assuming* one of them is right.

In simulation the true value is available in closed form: the environment
exposes the noise-free expected reward of every action, so the target policy's
value on the logged context distribution can be computed exactly (see
``src.evaluation.ope.true_policy_value``).  That turns "which estimator do I
trust" from a matter of opinion into a measurement.

What this script measures, over R independent replications:

- **Bias** — mean signed error against the true policy value
- **RMSE** — total error magnitude
- **CI coverage** — how often the reported 95% interval actually contains the
  truth.  An estimator with small bias but 40% coverage is more dangerous than
  one with visible bias and honest intervals, because it will be trusted.
- **Effective sample size** — how much of the log each estimator is really using

and how all of it varies with (a) how much the logging policy explored, and
(b) how large the log is.

Usage
-----
    python scripts/run_ope_study.py --replications 10 --log-size 4000
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

from src.evaluation.ope import (  # noqa: E402
    evaluate_all,
    fit_reward_model,
    true_policy_value,
)
from src.evaluation.policies import (  # noqa: E402
    EpsilonGreedyPolicy,
    NeuralLinearPolicy,
    RulePolicy,
)
from src.evaluation.runner import collect_logged_data, run_policy  # noqa: E402
from src.feature_store.transform import FEATURE_DIM  # noqa: E402
from src.recommendation.action_space import ActionSpace  # noqa: E402

ESTIMATORS = ["IPS", "SNIPS", "CIPS(M=10)", "DM", "DR"]


def train_target_policy(seed: int, episodes: int, users: int) -> NeuralLinearPolicy:
    """
    Train the deployment candidate online, then freeze it.

    The target policy has to be fixed before the log is evaluated: evaluating a
    policy that is still learning would mean the thing being estimated changes
    while it is being estimated.
    """
    policy = NeuralLinearPolicy(18, FEATURE_DIM, seed=seed)
    run_policy(
        policy,
        n_episodes=episodes,
        n_users=users,
        seed=seed,
        record_scores=False,
    )
    return policy


def one_replication(
    seed: int,
    log_size: int,
    epsilon: float,
    train_episodes: int,
    users: int,
    mc_samples: int,
) -> Dict[str, dict]:
    """Train a target, collect a log, and score every estimator against truth."""
    space = ActionSpace()

    target = train_target_policy(seed, train_episodes, users)

    # Logging policy: the production rule-based system with forced exploration.
    # Its propensities are exact and strictly positive on every gated action,
    # which is the identification condition the IPS family needs.
    behaviour = EpsilonGreedyPolicy(
        base=RulePolicy(space.get_action_count()), epsilon=epsilon, seed=seed
    )

    batch = collect_logged_data(
        behaviour,
        n_episodes=log_size,
        n_users=users,
        seed=seed + 10_000,
        action_space=space,
        learn=False,
    )

    target_probs = target.propensities_batch(
        batch.contexts, batch.allowed_masks, n_samples=mc_samples
    )

    v_true = true_policy_value(batch, target_probs)
    v_behaviour = true_policy_value(
        batch,
        behaviour.propensities_batch(batch.contexts, batch.allowed_masks),
    )

    q_hat = fit_reward_model(batch, seed=seed)
    results = evaluate_all(batch, target_probs, q_hat=q_hat, seed=seed)

    out: Dict[str, dict] = {
        "_truth": {
            "target_value": v_true,
            "behaviour_value": v_behaviour,
            "true_lift_pct": (v_true - v_behaviour) / abs(v_behaviour) * 100,
        }
    }
    for r in results:
        lo, hi = r.ci_95
        out[r.estimator] = {
            "value": r.value,
            "error": r.value - v_true,
            "abs_error": abs(r.value - v_true),
            "std_error": r.std_error,
            "ci_low": lo,
            "ci_high": hi,
            "covers_truth": bool(lo <= v_true <= hi),
            "ess_fraction": r.diagnostics.get("ess_fraction", float("nan")),
            "max_weight": r.diagnostics.get("max_weight", float("nan")),
        }
    return out


def summarise(reps: List[Dict[str, dict]]) -> Dict[str, dict]:
    """Aggregate replications into bias / RMSE / coverage per estimator."""
    summary: Dict[str, dict] = {}
    truths = np.array([r["_truth"]["target_value"] for r in reps])
    summary["_truth"] = {
        "target_value_mean": float(np.mean(truths)),
        "target_value_std": float(np.std(truths, ddof=1)) if len(truths) > 1 else 0.0,
        "behaviour_value_mean": float(
            np.mean([r["_truth"]["behaviour_value"] for r in reps])
        ),
        "true_lift_pct_mean": float(
            np.mean([r["_truth"]["true_lift_pct"] for r in reps])
        ),
    }

    for est in ESTIMATORS:
        errors = np.array([r[est]["error"] for r in reps])
        values = np.array([r[est]["value"] for r in reps])
        covers = np.array([r[est]["covers_truth"] for r in reps], dtype=float)
        ess = np.array([r[est]["ess_fraction"] for r in reps], dtype=float)
        # The Direct Method never touches the importance weights, so it has no
        # effective sample size — it uses every row equally.  Reported as NaN
        # rather than 100% so the table cannot be misread as "DM is efficient".
        mean_ess = float(np.nanmean(ess)) if not np.all(np.isnan(ess)) else float("nan")
        summary[est] = {
            "bias": float(np.mean(errors)),
            "abs_bias": float(np.abs(np.mean(errors))),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "ci_coverage": float(np.mean(covers)),
            "mean_ess_fraction": mean_ess,
        }
    return summary


def _print_summary(title: str, summary: Dict[str, dict]) -> None:
    t = summary["_truth"]
    print()
    print(f"  {title}")
    print("  " + "-" * 88)
    print(
        f"  True target value: {t['target_value_mean']:+.4f} "
        f"(± {t['target_value_std']:.4f})   "
        f"logging policy: {t['behaviour_value_mean']:+.4f}   "
        f"true lift: {t['true_lift_pct_mean']:+.1f}%"
    )
    print(
        f"  {'Estimator':<12}{'Bias':>10}{'Std':>10}{'RMSE':>10}"
        f"{'95% CI cov.':>14}{'ESS frac':>11}"
    )
    for est in ESTIMATORS:
        s = summary[est]
        ess = s["mean_ess_fraction"]
        ess_txt = "n/a" if np.isnan(ess) else f"{ess * 100:.1f}%"
        print(
            f"  {est:<12}{s['bias']:>+10.4f}{s['std']:>10.4f}{s['rmse']:>10.4f}"
            f"{s['ci_coverage'] * 100:>13.0f}%{ess_txt:>11}"
        )


def _plot(all_summaries: Dict[str, Dict[str, dict]], out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = list(all_summaries.keys())
    x = np.arange(len(ESTIMATORS))
    width = 0.8 / max(len(labels), 1)
    colors = ["#3498db", "#27ae60", "#e67e22", "#9b59b6"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for i, label in enumerate(labels):
        s = all_summaries[label]
        offset = (i - (len(labels) - 1) / 2) * width

        axes[0].bar(
            x + offset,
            [s[e]["bias"] for e in ESTIMATORS],
            width,
            label=label,
            color=colors[i % len(colors)],
        )
        axes[1].bar(
            x + offset,
            [s[e]["rmse"] for e in ESTIMATORS],
            width,
            label=label,
            color=colors[i % len(colors)],
        )
        axes[2].bar(
            x + offset,
            [s[e]["ci_coverage"] * 100 for e in ESTIMATORS],
            width,
            label=label,
            color=colors[i % len(colors)],
        )

    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Bias vs. simulator ground truth\n(0 = unbiased)")
    axes[0].set_ylabel("Estimate − true value")

    axes[1].set_title("RMSE\n(lower is better)")
    axes[1].set_ylabel("RMSE")

    axes[2].axhline(95, color="red", linestyle="--", linewidth=1.2, label="nominal 95%")
    axes[2].set_title("95% CI coverage\n(should be ~95%)")
    axes[2].set_ylabel("% of replications containing truth")
    axes[2].set_ylim(0, 105)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ESTIMATORS, rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Off-policy estimators scored against simulator ground truth",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="OPE estimator calibration study")
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--log-size", type=int, default=4000)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--mc-samples", type=int, default=200)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.3, 0.8],
        help="logging-policy exploration rates to compare",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("=" * 92)
    print("  Off-Policy Evaluation — Estimator Calibration Study")
    print("=" * 92)
    print(
        f"\n  Replications: {args.replications}   Log size: {args.log_size}   "
        f"Target: NeuralLinear trained {args.train_episodes} episodes"
    )
    print(f"  Logging policy: epsilon-greedy over rules, eps in {args.epsilons}\n")

    t0 = time.time()
    all_summaries: Dict[str, Dict[str, dict]] = {}
    all_reps: Dict[str, list] = {}

    for eps in args.epsilons:
        label = f"eps={eps:g}"
        print(f"  --- logging policy {label} ---")
        reps = []
        for r in range(args.replications):
            rep = one_replication(
                seed=r + 1,
                log_size=args.log_size,
                epsilon=eps,
                train_episodes=args.train_episodes,
                users=args.users,
                mc_samples=args.mc_samples,
            )
            reps.append(rep)
            print(
                f"    rep {r + 1:>2}: true={rep['_truth']['target_value']:+.4f}  "
                + "  ".join(
                    f"{e}={rep[e]['value']:+.4f}" for e in ("IPS", "SNIPS", "DM", "DR")
                )
            )
        all_reps[label] = reps
        all_summaries[label] = summarise(reps)
        _print_summary(f"Summary — logging policy {label}", all_summaries[label])

    payload = {
        "config": {
            "replications": args.replications,
            "log_size": args.log_size,
            "train_episodes": args.train_episodes,
            "users": args.users,
            "mc_samples": args.mc_samples,
            "epsilons": args.epsilons,
            "feature_dim": FEATURE_DIM,
            "note": (
                "Ground truth available only because the environment is simulated. "
                "See src/simulation/fitness_env.py."
            ),
        },
        "summary": all_summaries,
        "replications_detail": all_reps,
    }
    out_path = Path(__file__).parent / "ope_study_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  Raw results -> {out_path}")

    if not args.no_plot:
        plot_path = Path(__file__).parent.parent / "docs" / "ope_calibration.png"
        if _plot(all_summaries, plot_path):
            print(f"  Plot        -> {plot_path}")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
