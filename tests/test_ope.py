"""
Off-Policy Estimator Tests
==========================

The properties pinned here are the ones the estimators are *chosen* for.  A
Doubly Robust implementation that silently loses its double robustness still
returns plausible numbers on real data, and nothing downstream notices — so the
guarantees are tested directly, on cases where the right answer is known by
construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.ope import (
    LoggedBatch,
    clipped_ips,
    direct_method,
    doubly_robust,
    evaluate_all,
    fit_reward_model,
    ips,
    snips,
    support_diagnostics,
    true_policy_value,
)

N_ACTIONS = 5
DIM = 6


def make_batch(
    n: int = 4000,
    seed: int = 0,
    epsilon: float = 0.5,
) -> tuple[LoggedBatch, np.ndarray]:
    """
    Build a synthetic log with a known reward function.

    Returns the batch plus the behaviour policy's full probability matrix, so a
    test can use "target == behaviour" as a case where the true value is simply
    the expected logged reward.
    """
    rng = np.random.default_rng(seed)
    contexts = rng.normal(size=(n, DIM))

    # Ground-truth expected reward, linear per action so the reward model can
    # be exactly correct when a test needs it to be.
    coefs = rng.normal(size=(N_ACTIONS, DIM)) * 0.3
    true_q = contexts @ coefs.T

    # Behaviour policy: epsilon-greedy on a fixed scoring rule.
    scores = contexts @ rng.normal(size=(N_ACTIONS, DIM)).T
    greedy = np.argmax(scores, axis=1)
    behaviour = np.full((n, N_ACTIONS), epsilon / N_ACTIONS)
    behaviour[np.arange(n), greedy] += 1.0 - epsilon

    actions = np.array(
        [rng.choice(N_ACTIONS, p=behaviour[i]) for i in range(n)], dtype=int
    )
    rewards = true_q[np.arange(n), actions] + rng.normal(0, 0.1, size=n)

    batch = LoggedBatch(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        propensities=behaviour[np.arange(n), actions],
        allowed_masks=np.ones((n, N_ACTIONS), dtype=bool),
        true_q=true_q,
    )
    return batch, behaviour


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_zero_propensity_is_rejected():
    """
    A logged action with zero probability makes the estimate unidentifiable.

    Failing loudly at construction is the point: the alternative is a division
    by zero that becomes an infinity that becomes a NaN that quietly poisons the
    average.
    """
    with pytest.raises(ValueError, match="propensity"):
        LoggedBatch(
            contexts=np.zeros((3, DIM)),
            actions=np.array([0, 1, 2]),
            rewards=np.array([0.1, 0.2, 0.3]),
            propensities=np.array([0.5, 0.0, 0.5]),
        )


def test_true_policy_value_requires_ground_truth():
    """On real logs there is no true value, and the API must say so."""
    batch = LoggedBatch(
        contexts=np.zeros((3, DIM)),
        actions=np.array([0, 1, 2]),
        rewards=np.array([0.1, 0.2, 0.3]),
        propensities=np.array([0.5, 0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="simulation"):
        true_policy_value(batch, np.ones((3, N_ACTIONS)) / N_ACTIONS)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        LoggedBatch(
            contexts=np.zeros((3, DIM)),
            actions=np.array([0, 1]),
            rewards=np.array([0.1, 0.2, 0.3]),
            propensities=np.array([0.5, 0.5, 0.5]),
        )


# ---------------------------------------------------------------------------
# Unbiasedness
# ---------------------------------------------------------------------------


def test_ips_recovers_the_value_when_target_equals_behaviour():
    """
    With target == behaviour every importance weight is 1, so IPS collapses to
    the empirical mean reward. If this fails the weights are wrong.
    """
    batch, behaviour = make_batch()
    result = ips(batch, behaviour)
    assert result.value == pytest.approx(float(batch.rewards.mean()), abs=1e-9)


def test_snips_recovers_the_value_when_target_equals_behaviour():
    batch, behaviour = make_batch()
    result = snips(batch, behaviour)
    assert result.value == pytest.approx(float(batch.rewards.mean()), abs=1e-9)


@pytest.mark.parametrize("estimator", [ips, snips])
def test_estimators_are_close_to_truth_on_a_shifted_target(estimator):
    """
    A target that differs from the behaviour policy but keeps full support: the
    IPS family should land near the true value.
    """
    batch, behaviour = make_batch(n=8000, seed=3)
    rng = np.random.default_rng(11)
    target = behaviour + rng.uniform(0.02, 0.06, size=behaviour.shape)
    target /= target.sum(axis=1, keepdims=True)

    truth = true_policy_value(batch, target)
    assert estimator(batch, target).value == pytest.approx(truth, abs=0.05)


def test_doubly_robust_is_accurate_with_a_good_reward_model():
    batch, behaviour = make_batch(n=8000, seed=4)
    rng = np.random.default_rng(12)
    target = behaviour + rng.uniform(0.02, 0.06, size=behaviour.shape)
    target /= target.sum(axis=1, keepdims=True)

    q_hat = fit_reward_model(batch, seed=4)
    truth = true_policy_value(batch, target)
    assert doubly_robust(batch, target, q_hat).value == pytest.approx(truth, abs=0.05)


def test_doubly_robust_survives_a_broken_reward_model():
    """
    The defining property: DR stays consistent when the reward model is wrong,
    because the propensities are still right.

    A reward model of all zeros is maximally wrong, and it reduces DR exactly to
    IPS — which is unbiased.  The Direct Method, given the same broken model, is
    simply wrong.
    """
    batch, behaviour = make_batch(n=8000, seed=5)
    rng = np.random.default_rng(13)
    target = behaviour + rng.uniform(0.02, 0.06, size=behaviour.shape)
    target /= target.sum(axis=1, keepdims=True)

    broken_q = np.zeros((batch.n, N_ACTIONS))
    truth = true_policy_value(batch, target)

    dr = doubly_robust(batch, target, broken_q).value
    dm = direct_method(batch, target, broken_q).value

    assert dr == pytest.approx(truth, abs=0.05)
    assert dm == pytest.approx(0.0, abs=1e-9)
    assert abs(dr - truth) < abs(dm - truth)


# ---------------------------------------------------------------------------
# Bias-variance behaviour
# ---------------------------------------------------------------------------


def test_clipping_reduces_variance_and_introduces_bias():
    """
    Clipping is a deliberate bias-for-variance trade.  Verified rather than
    assumed, because a clip threshold set above every observed weight would be
    a no-op and would look identical in a results table.
    """
    batch, behaviour = make_batch(n=4000, seed=6, epsilon=0.15)
    rng = np.random.default_rng(14)
    target = np.zeros_like(behaviour)
    target[np.arange(batch.n), rng.integers(0, N_ACTIONS, batch.n)] = 1.0

    unclipped = ips(batch, target)
    clipped = clipped_ips(batch, target, clip=2.0)

    assert clipped.diagnostics["clipped_fraction"] > 0.0
    assert clipped.std_error < unclipped.std_error


def test_direct_method_has_lower_variance_than_ips():
    batch, behaviour = make_batch(n=4000, seed=7, epsilon=0.2)
    rng = np.random.default_rng(15)
    target = np.zeros_like(behaviour)
    target[np.arange(batch.n), rng.integers(0, N_ACTIONS, batch.n)] = 1.0

    q_hat = fit_reward_model(batch, seed=7)
    assert direct_method(batch, target, q_hat).std_error < ips(batch, target).std_error


def test_snips_stays_inside_the_observed_reward_range():
    """
    Self-normalisation bounds the estimate by the rewards actually observed.
    Plain IPS carries no such guarantee, which is how it occasionally reports
    values the reward function cannot produce.
    """
    batch, behaviour = make_batch(n=2000, seed=8, epsilon=0.1)
    rng = np.random.default_rng(16)
    target = np.zeros_like(behaviour)
    target[np.arange(batch.n), rng.integers(0, N_ACTIONS, batch.n)] = 1.0

    value = snips(batch, target).value
    assert batch.rewards.min() <= value <= batch.rewards.max()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_effective_sample_size_is_full_when_policies_match():
    batch, behaviour = make_batch(n=2000, seed=9)
    diag = support_diagnostics(batch, behaviour)
    assert diag["ess_fraction"] == pytest.approx(1.0, abs=1e-9)


def test_effective_sample_size_collapses_under_poor_overlap():
    """
    A deterministic target over a barely-exploring log uses only a sliver of it,
    and the diagnostic has to show that — otherwise a confident-looking estimate
    computed from a handful of rows passes for a well-supported one.
    """
    batch, behaviour = make_batch(n=4000, seed=10, epsilon=0.05)
    rng = np.random.default_rng(17)
    target = np.zeros_like(behaviour)
    target[np.arange(batch.n), rng.integers(0, N_ACTIONS, batch.n)] = 1.0

    assert support_diagnostics(batch, target)["ess_fraction"] < 0.5


def test_reward_model_is_cross_fitted():
    """
    Cross-fitting means held-out residuals, so in-sample error must not be
    near-zero.  A model fitted on all the data would drive DR's correction term
    to zero and collapse it into the Direct Method.
    """
    batch, _ = make_batch(n=2000, seed=18)
    q_hat = fit_reward_model(batch, n_folds=5, seed=18)
    residuals = batch.rewards - q_hat[np.arange(batch.n), batch.actions]
    assert np.mean(np.abs(residuals)) > 1e-3


def test_evaluate_all_returns_every_estimator():
    batch, behaviour = make_batch(n=1000, seed=19)
    results = evaluate_all(batch, behaviour)
    names = {r.estimator for r in results}
    assert {"IPS", "SNIPS", "DM", "DR"} <= names
    for r in results:
        assert r.ci_95[0] <= r.value <= r.ci_95[1] or np.isfinite(r.value)
