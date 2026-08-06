"""
NeuralLinear and Linear Thompson Sampling Tests
===============================================

Covers the properties that make these policies bandits rather than just
regressors: calibrated posteriors, exploration that respects the safety gate,
and correct handling of the representation changing underneath the posteriors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recommendation.bayes_linear import NIGPosteriorSet
from src.recommendation.linear_ts import LinearThompsonSampling
from src.recommendation.neural_linear import NeuralLinearBandit, NeuralLinearConfig

N_ACTIONS = 6
DIM = 8


def _fast_config(**overrides) -> NeuralLinearConfig:
    """Small/quick settings so the suite stays fast."""
    base = dict(
        hidden_sizes=(32, 16),
        repr_dim=16,
        train_every=100,
        train_epochs=8,
        min_train_size=32,
    )
    base.update(overrides)
    return NeuralLinearConfig(**base)


# ---------------------------------------------------------------------------
# Posterior mechanics
# ---------------------------------------------------------------------------


def test_posterior_mean_converges_to_the_true_coefficients():
    """With enough data the posterior mean must recover the generating weights."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=DIM)
    posteriors = NIGPosteriorSet(n_actions=1, dim=DIM, seed=0)

    for _ in range(3000):
        x = rng.normal(size=DIM)
        posteriors.update(0, x, float(x @ truth + rng.normal(0, 0.05)))

    assert np.allclose(posteriors.mu[0], truth, atol=0.05)


def test_posterior_uncertainty_shrinks_with_data():
    """
    Repeated draws must become more alike as evidence accumulates.  A posterior
    whose spread does not shrink makes Thompson Sampling explore forever.
    """
    rng = np.random.default_rng(1)
    truth = rng.normal(size=DIM)
    posteriors = NIGPosteriorSet(n_actions=1, dim=DIM, seed=1)

    early = np.std([posteriors.sample_weights(0) for _ in range(200)], axis=0).mean()
    for _ in range(2000):
        x = rng.normal(size=DIM)
        posteriors.update(0, x, float(x @ truth + rng.normal(0, 0.05)))
    late = np.std([posteriors.sample_weights(0) for _ in range(200)], axis=0).mean()

    assert late < early / 2


def test_sherman_morrison_matches_an_explicit_inverse():
    """
    The incremental inverse update is an optimisation; it must be numerically
    identical to recomputing the inverse from scratch.
    """
    rng = np.random.default_rng(2)
    posteriors = NIGPosteriorSet(n_actions=1, dim=DIM, lambda_prior=1.0, seed=2)
    for _ in range(200):
        posteriors.update(0, rng.normal(size=DIM), float(rng.normal()))

    assert np.allclose(posteriors.A_inv[0], np.linalg.inv(posteriors.A[0]), atol=1e-8)


def test_invalid_priors_are_rejected():
    with pytest.raises(ValueError, match="alpha_prior"):
        NIGPosteriorSet(n_actions=2, dim=DIM, alpha_prior=1.0)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------


def _nonlinear_reward(rng, weights):
    def reward(x: np.ndarray, action: int) -> float:
        ideal = 3.0 / (1.0 + np.exp(-float(x @ weights)))
        return float(
            np.clip(0.8 - 0.25 * (action - ideal) ** 2 + rng.normal(0, 0.05), -1, 1)
        )

    return reward


def test_neural_linear_learns_a_nonlinear_reward():
    rng = np.random.default_rng(3)
    reward = _nonlinear_reward(rng, rng.normal(size=DIM))
    bandit = NeuralLinearBandit(N_ACTIONS, DIM, _fast_config(), seed=3)

    history = []
    for _ in range(2500):
        x = rng.normal(size=DIM)
        a = bandit.select_action(x)
        r = reward(x, a)
        bandit.update(x, a, r)
        history.append(r)

    assert np.mean(history[-500:]) > np.mean(history[:500]) + 0.1


def test_neural_linear_beats_linear_ts_on_a_nonlinear_reward():
    """
    The reason NeuralLinear exists.  Same posterior machinery, same exploration;
    only the representation differs, so a gap here is attributable to learning
    the representation.
    """
    weights = np.random.default_rng(4).normal(size=DIM)

    def run(agent, seed):
        rng = np.random.default_rng(seed)
        reward = _nonlinear_reward(rng, weights)
        out = []
        for _ in range(2500):
            x = rng.normal(size=DIM)
            a = agent.select_action(x)
            r = reward(x, a)
            agent.update(x, a, r)
            out.append(r)
        return float(np.mean(out[-500:]))

    linear = run(LinearThompsonSampling(N_ACTIONS, DIM, seed=5), 5)
    neural = run(NeuralLinearBandit(N_ACTIONS, DIM, _fast_config(), seed=5), 5)
    assert neural > linear


# ---------------------------------------------------------------------------
# Safety-gate integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_factory",
    [
        lambda: LinearThompsonSampling(N_ACTIONS, DIM, seed=6),
        lambda: NeuralLinearBandit(N_ACTIONS, DIM, _fast_config(), seed=6),
    ],
)
def test_selection_never_leaves_the_allowed_set(agent_factory):
    """
    Exploration must stay inside the gated set.  Sampling freely and masking
    afterwards would occasionally surface an unsafe action.
    """
    agent = agent_factory()
    rng = np.random.default_rng(7)
    allowed = [0, 2, 5]
    for _ in range(300):
        x = rng.normal(size=DIM)
        action = agent.select_action(x, allowed)
        assert action in allowed
        agent.update(x, action, float(rng.normal()))


def test_expected_rewards_are_negative_infinity_outside_the_gate():
    agent = LinearThompsonSampling(N_ACTIONS, DIM, seed=8)
    scores = agent.expected_rewards(np.ones(DIM), [1, 3])
    assert np.isfinite(scores[1]) and np.isfinite(scores[3])
    assert np.isneginf(scores[0]) and np.isneginf(scores[2])


# ---------------------------------------------------------------------------
# Propensities
# ---------------------------------------------------------------------------


def test_action_probabilities_form_a_distribution_over_allowed_actions():
    agent = LinearThompsonSampling(N_ACTIONS, DIM, seed=9)
    probs = agent.action_probabilities(np.ones(DIM), [0, 2, 4], n_samples=200)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[1] == 0.0 and probs[3] == 0.0 and probs[5] == 0.0


def test_batched_probabilities_agree_with_the_per_row_version():
    """
    The vectorised estimator is an optimisation for off-policy evaluation; it
    has to estimate the same distribution the per-row version does.
    """
    rng = np.random.default_rng(10)
    agent = LinearThompsonSampling(N_ACTIONS, DIM, seed=10)
    for _ in range(400):
        x = rng.normal(size=DIM)
        a = agent.select_action(x)
        agent.update(x, a, float(x.sum() * 0.1 + rng.normal(0, 0.1)))

    contexts = rng.normal(size=(40, DIM))
    masks = np.ones((40, N_ACTIONS), dtype=bool)

    batched = agent.action_probabilities_batch(contexts, masks, n_samples=800)
    per_row = np.vstack(
        [
            agent.action_probabilities(c, list(range(N_ACTIONS)), n_samples=800)
            for c in contexts
        ]
    )

    assert np.allclose(batched.sum(axis=1), 1.0)
    # Monte-Carlo estimates, so agreement is statistical, not exact.
    assert np.mean(np.abs(batched - per_row)) < 0.06


def test_batched_probabilities_respect_masks():
    agent = LinearThompsonSampling(N_ACTIONS, DIM, seed=11)
    masks = np.zeros((5, N_ACTIONS), dtype=bool)
    masks[:, [1, 4]] = True
    probs = agent.action_probabilities_batch(np.ones((5, DIM)), masks, n_samples=100)
    assert np.all(probs[:, [0, 2, 3, 5]] == 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Representation staleness and serving
# ---------------------------------------------------------------------------


def test_posteriors_are_rebuilt_after_an_encoder_retrain():
    """
    After the encoder changes, posterior counts must equal the replay buffer
    size — proof the rebuild ran.  Skipping it leaves posteriors expressed in a
    representation space that no longer exists.
    """
    rng = np.random.default_rng(12)
    bandit = NeuralLinearBandit(N_ACTIONS, DIM, _fast_config(train_every=50), seed=12)
    for _ in range(200):
        x = rng.normal(size=DIM)
        a = bandit.select_action(x)
        bandit.update(x, a, float(rng.normal()))

    assert bandit.statistics()["n_retrains"] > 0
    assert int(bandit.action_counts.sum()) == len(bandit._buf_x)


def test_frozen_encoder_never_retrains_and_preserves_prior_counts():
    """
    The serving configuration.  A frozen encoder must leave counts monotonically
    increasing — the bug this guards against wiped a loaded checkpoint's
    posteriors on the first retrain, because a fresh process has an empty replay
    buffer to rebuild from.
    """
    rng = np.random.default_rng(13)
    bandit = NeuralLinearBandit(
        N_ACTIONS, DIM, _fast_config(freeze_encoder=True, train_every=50), seed=13
    )

    counts = [int(bandit.action_counts.sum())]
    for _ in range(300):
        x = rng.normal(size=DIM)
        a = bandit.select_action(x)
        bandit.update(x, a, float(rng.normal()))
        counts.append(int(bandit.action_counts.sum()))

    assert bandit.statistics()["n_retrains"] == 0
    assert counts == sorted(counts)
    assert counts[-1] == counts[0] + 300


def test_state_dict_round_trip_preserves_behaviour():
    """A restored checkpoint must score identically to the model it came from."""
    rng = np.random.default_rng(14)
    original = NeuralLinearBandit(N_ACTIONS, DIM, _fast_config(), seed=14)
    for _ in range(300):
        x = rng.normal(size=DIM)
        a = original.select_action(x)
        original.update(x, a, float(x.sum() * 0.1 + rng.normal(0, 0.1)))

    restored = NeuralLinearBandit(N_ACTIONS, DIM, _fast_config(), seed=99)
    restored.load_state_dict(original.state_dict())

    probe = rng.normal(size=(10, DIM))
    for x in probe:
        assert np.allclose(
            original.expected_rewards(x), restored.expected_rewards(x), atol=1e-9
        )
    assert np.array_equal(original.action_counts, restored.action_counts)


def test_wrong_context_dimension_is_rejected():
    bandit = NeuralLinearBandit(N_ACTIONS, DIM, _fast_config(), seed=15)
    with pytest.raises(ValueError, match="dim"):
        bandit.update(np.ones(DIM + 3), 0, 0.5)
