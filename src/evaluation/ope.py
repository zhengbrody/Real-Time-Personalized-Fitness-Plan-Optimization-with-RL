"""
Off-Policy Evaluation
=====================

Estimating the value of a *target* policy from data logged under a different
*behaviour* policy.

Why this matters here
---------------------
The online benchmark answers "how well does this policy do when it runs".  That
is the wrong question in production: you cannot run an unproven recommender on
real users to find out whether it is better.  What you can do is take the logs
your current policy already produced and ask what a candidate policy *would*
have earned.  Every estimator below answers that question with a different
bias-variance trade-off:

===========  ==========================================  ================
Estimator    Unbiased when...                            Variance
===========  ==========================================  ================
IPS          propensities are correct                    high
SNIPS        (asymptotically) propensities are correct   medium
CIPS         never — clipping trades bias for variance   low
DM           the reward model is correct                 low
DR           *either* propensities *or* reward model      medium
===========  ==========================================  ================

Doubly Robust is the one to reach for: it only needs one of the two nuisance
models to be right, and in practice the propensities are known exactly (the
logging policy is ours) while the reward model is the shaky one.

The identification condition
----------------------------
Every one of these needs **common support**: wherever the target policy would
act, the behaviour policy must have had positive probability of acting the same
way.  If the logging policy never tried HIGH intensity on a fatigued user, no
amount of estimator sophistication can tell you what would have happened.
:func:`support_diagnostics` reports the violation mass explicitly rather than
letting it silently bias the estimate — a zero-propensity action does not raise
an error, it just quietly disappears from the average.

Validation
----------
On real logs the true value is unknowable, so estimator quality is a matter of
faith.  In simulation it is not: ``true_q`` carries the simulator's noise-free
expected reward for *every* action, so :func:`true_policy_value` computes the
exact target value on the logged context distribution.  That is what
``scripts/run_ope_study.py`` uses to measure each estimator's actual bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "LoggedBatch",
    "OPEResult",
    "ips",
    "snips",
    "clipped_ips",
    "direct_method",
    "doubly_robust",
    "true_policy_value",
    "support_diagnostics",
    "fit_reward_model",
    "evaluate_all",
]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class LoggedBatch:
    """
    One batch of logged bandit feedback.

    Attributes
    ----------
    contexts
        ``(n, d)`` feature vectors.
    actions
        ``(n,)`` action taken by the behaviour policy.
    rewards
        ``(n,)`` observed reward for the taken action only.  The counterfactual
        rewards are, by construction, not observed — that is the whole problem.
    propensities
        ``(n,)`` behaviour-policy probability of the taken action, ``mu(a_i|x_i)``.
    allowed_masks
        ``(n, K)`` boolean; which actions the safety gate permitted.  Estimators
        need this because the target policy's support is also gate-restricted.
    true_q
        ``(n, K)`` simulator ground truth, or ``None`` for real logs.  Present
        only in simulation, used exclusively for validating the estimators.
    """

    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    allowed_masks: Optional[np.ndarray] = None
    true_q: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n = len(self.actions)
        if not (len(self.rewards) == len(self.propensities) == n):
            raise ValueError(
                "actions, rewards and propensities must be the same length"
            )
        if self.contexts.shape[0] != n:
            raise ValueError("contexts must have one row per logged action")
        if np.any(self.propensities <= 0.0):
            raise ValueError(
                "zero or negative logging propensity: the batch is unusable for "
                "inverse-propensity estimation. Log the propensity at decision "
                "time rather than reconstructing it later."
            )

    @property
    def n(self) -> int:
        return len(self.actions)

    @property
    def n_actions(self) -> int:
        if self.allowed_masks is not None:
            return self.allowed_masks.shape[1]
        if self.true_q is not None:
            return self.true_q.shape[1]
        return int(self.actions.max()) + 1


@dataclass
class OPEResult:
    """An estimate with its uncertainty and the diagnostics behind it."""

    estimator: str
    value: float
    std_error: float
    ci_95: Sequence[float]
    diagnostics: Dict[str, float]

    def __str__(self) -> str:  # pragma: no cover - display helper
        lo, hi = self.ci_95
        return (
            f"{self.estimator:<10} {self.value:+.4f} "
            f"± {self.std_error:.4f}  [{lo:+.4f}, {hi:+.4f}]"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _importance_weights(batch: LoggedBatch, target_probs: np.ndarray) -> np.ndarray:
    """``w_i = pi(a_i | x_i) / mu(a_i | x_i)``."""
    idx = np.arange(batch.n)
    pi_a = target_probs[idx, batch.actions]
    return pi_a / batch.propensities


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, seed: int = 0, statistic="mean"
) -> tuple:
    """
    Percentile bootstrap over the per-sample contributions.

    Used instead of a normal approximation because importance-weighted
    estimates are heavy-tailed — a handful of large weights dominate, and the
    sampling distribution is visibly skewed at realistic sample sizes.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = float(np.mean(values[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def support_diagnostics(
    batch: LoggedBatch, target_probs: np.ndarray
) -> Dict[str, float]:
    """
    How badly the common-support condition is violated, and how concentrated
    the importance weights are.

    ``effective_sample_size`` is the Kish ESS: the number of *equally weighted*
    samples that would carry the same information.  When it collapses to a
    small fraction of ``n``, the IPS family is effectively averaging over a
    handful of observations regardless of how large the log is, and its
    confidence interval should not be believed.
    """
    w = _importance_weights(batch, target_probs)
    ess = float((w.sum() ** 2) / max(np.sum(w**2), 1e-12))

    # Mass the target places on actions the behaviour policy could never take.
    violation = 0.0
    if batch.allowed_masks is not None:
        # Any target mass outside the gated set is a modelling error, not a
        # support violation, so check that separately.
        outside = target_probs * (~batch.allowed_masks)
        violation = float(np.mean(outside.sum(axis=1)))

    return {
        "effective_sample_size": ess,
        "ess_fraction": ess / batch.n,
        "max_weight": float(w.max()),
        "mean_weight": float(w.mean()),
        "weight_p99": float(np.percentile(w, 99)),
        "target_mass_outside_gate": violation,
    }


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def ips(batch: LoggedBatch, target_probs: np.ndarray, seed: int = 0) -> OPEResult:
    """
    Inverse Propensity Scoring.

    ``V = mean_i [ w_i * r_i ]`` — unbiased under common support, and the
    highest-variance estimator in the family because nothing bounds ``w_i``.
    """
    w = _importance_weights(batch, target_probs)
    contributions = w * batch.rewards
    value = float(np.mean(contributions))
    se = float(np.std(contributions, ddof=1) / np.sqrt(batch.n))
    return OPEResult(
        estimator="IPS",
        value=value,
        std_error=se,
        ci_95=_bootstrap_ci(contributions, seed=seed),
        diagnostics=support_diagnostics(batch, target_probs),
    )


def snips(batch: LoggedBatch, target_probs: np.ndarray, seed: int = 0) -> OPEResult:
    """
    Self-Normalised IPS: ``V = sum_i w_i r_i / sum_i w_i``.

    Dividing by the realised weight mass rather than by ``n`` removes the
    scale error that dominates IPS when the weights happen to sum to much more
    or less than ``n``.  The price is a small finite-sample bias; the payoff is
    that the estimate is bounded by the observed reward range, so it cannot
    return the physically impossible values IPS sometimes does.
    """
    w = _importance_weights(batch, target_probs)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        raise ValueError("importance weights sum to zero; no overlap with the log")
    value = float(np.sum(w * batch.rewards) / w_sum)

    # Delta-method standard error for a ratio estimator.
    residual = w * (batch.rewards - value)
    se = float(np.sqrt(np.sum(residual**2)) / w_sum)

    rng = np.random.default_rng(seed)
    boots = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, batch.n, size=batch.n)
        wb = w[idx]
        boots[b] = float(np.sum(wb * batch.rewards[idx]) / max(np.sum(wb), 1e-12))

    return OPEResult(
        estimator="SNIPS",
        value=value,
        std_error=se,
        ci_95=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        diagnostics=support_diagnostics(batch, target_probs),
    )


def clipped_ips(
    batch: LoggedBatch,
    target_probs: np.ndarray,
    clip: float = 10.0,
    seed: int = 0,
) -> OPEResult:
    """
    IPS with importance weights capped at ``clip``.

    Deliberately biased downward — capping a weight discards exactly the rare,
    high-leverage observations that make IPS unbiased — in exchange for a large
    variance reduction.  Worth reporting alongside IPS because the gap between
    the two is a direct read on how much of the estimate rests on a few
    extreme weights.
    """
    w = np.minimum(_importance_weights(batch, target_probs), clip)
    contributions = w * batch.rewards
    value = float(np.mean(contributions))
    se = float(np.std(contributions, ddof=1) / np.sqrt(batch.n))
    diag = support_diagnostics(batch, target_probs)
    diag["clip"] = clip
    diag["clipped_fraction"] = float(
        np.mean(_importance_weights(batch, target_probs) > clip)
    )
    return OPEResult(
        estimator=f"CIPS(M={clip:g})",
        value=value,
        std_error=se,
        ci_95=_bootstrap_ci(contributions, seed=seed),
        diagnostics=diag,
    )


def direct_method(
    batch: LoggedBatch,
    target_probs: np.ndarray,
    q_hat: np.ndarray,
    seed: int = 0,
) -> OPEResult:
    """
    Direct Method: average the *modelled* reward under the target policy.

    ``V = mean_i sum_a pi(a|x_i) q_hat(x_i, a)``

    Ignores the logged rewards entirely once the model is fitted, so its
    variance is tiny and its bias is whatever the reward model gets wrong.  On
    logged data where the behaviour policy rarely tried some action, ``q_hat``
    for that action is an extrapolation, and DM will confidently report it.
    """
    per_sample = np.sum(target_probs * q_hat, axis=1)
    value = float(np.mean(per_sample))
    se = float(np.std(per_sample, ddof=1) / np.sqrt(batch.n))
    return OPEResult(
        estimator="DM",
        value=value,
        std_error=se,
        ci_95=_bootstrap_ci(per_sample, seed=seed),
        diagnostics={"model_free": 0.0},
    )


def doubly_robust(
    batch: LoggedBatch,
    target_probs: np.ndarray,
    q_hat: np.ndarray,
    seed: int = 0,
) -> OPEResult:
    """
    Doubly Robust: Direct Method plus an importance-weighted correction on the
    model's residual.

    ``V = mean_i [ sum_a pi(a|x_i) q_hat(x_i,a) + w_i (r_i - q_hat(x_i,a_i)) ]``

    Consistent if *either* nuisance model is right.  The correction term has
    the same ``w_i`` as IPS but multiplies the model residual instead of the
    raw reward, so when the reward model is decent the residuals are small and
    the variance stays far below IPS.
    """
    idx = np.arange(batch.n)
    w = _importance_weights(batch, target_probs)
    baseline = np.sum(target_probs * q_hat, axis=1)
    correction = w * (batch.rewards - q_hat[idx, batch.actions])
    per_sample = baseline + correction

    value = float(np.mean(per_sample))
    se = float(np.std(per_sample, ddof=1) / np.sqrt(batch.n))
    diag = support_diagnostics(batch, target_probs)
    diag["mean_abs_residual"] = float(
        np.mean(np.abs(batch.rewards - q_hat[idx, batch.actions]))
    )
    return OPEResult(
        estimator="DR",
        value=value,
        std_error=se,
        ci_95=_bootstrap_ci(per_sample, seed=seed),
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# Reward model for DM / DR
# ---------------------------------------------------------------------------


def fit_reward_model(
    batch: LoggedBatch,
    n_folds: int = 5,
    ridge: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Cross-fitted per-action ridge regression predicting reward from context.

    Returns ``q_hat`` of shape ``(n, K)``.

    **Cross-fitting matters.**  If the model were fitted on all the data and
    then used to predict the same data, its residuals would be optimistically
    small, and the Doubly Robust correction term — which is built out of those
    residuals — would be shrunk toward zero.  DR would silently degrade into
    the Direct Method and inherit its bias.  Fitting on K-1 folds and
    predicting the held-out fold keeps the residuals honest.

    A per-action linear model is deliberately simple: DM and DR are supposed to
    be evaluated under a *plausibly misspecified* reward model, since that is
    the situation on real logs.  The study script also reports what happens
    when the model is good.
    """
    n, d = batch.contexts.shape
    k = batch.n_actions
    q_hat = np.zeros((n, k))

    rng = np.random.default_rng(seed)
    fold_id = rng.integers(0, n_folds, size=n)

    eye = ridge * np.eye(d)
    for fold in range(n_folds):
        train = fold_id != fold
        test = ~train
        if not np.any(test):
            continue

        for action in range(k):
            rows = train & (batch.actions == action)
            if rows.sum() < 2:
                # Not enough data for this action in this fold: fall back to
                # the global mean reward rather than extrapolating from noise.
                q_hat[test, action] = (
                    float(np.mean(batch.rewards[train])) if train.any() else 0.0
                )
                continue
            X = batch.contexts[rows]
            y = batch.rewards[rows]
            coef = np.linalg.solve(X.T @ X + eye, X.T @ y)
            q_hat[test, action] = batch.contexts[test] @ coef

    return q_hat


# ---------------------------------------------------------------------------
# Ground truth (simulation only)
# ---------------------------------------------------------------------------


def true_policy_value(batch: LoggedBatch, target_probs: np.ndarray) -> float:
    """
    Exact target-policy value on the logged context distribution.

    ``V = mean_i sum_a pi(a|x_i) q(x_i, a)`` using the simulator's noise-free
    ``q``.  Only defined in simulation; this is the yardstick every estimator
    is scored against in the calibration study.
    """
    if batch.true_q is None:
        raise ValueError(
            "true_q is unavailable: ground-truth policy value only exists in "
            "simulation, not on real logged data."
        )
    return float(np.mean(np.sum(target_probs * batch.true_q, axis=1)))


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def evaluate_all(
    batch: LoggedBatch,
    target_probs: np.ndarray,
    q_hat: Optional[np.ndarray] = None,
    clip: float = 10.0,
    seed: int = 0,
) -> List[OPEResult]:
    """Run every estimator on the same batch and return them in one list."""
    if q_hat is None:
        q_hat = fit_reward_model(batch, seed=seed)
    return [
        ips(batch, target_probs, seed=seed),
        snips(batch, target_probs, seed=seed),
        clipped_ips(batch, target_probs, clip=clip, seed=seed),
        direct_method(batch, target_probs, q_hat, seed=seed),
        doubly_robust(batch, target_probs, q_hat, seed=seed),
    ]
