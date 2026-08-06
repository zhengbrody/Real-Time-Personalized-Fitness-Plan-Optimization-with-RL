"""
Simulator and Benchmark-Runner Tests
====================================

These two modules produce every number the project reports, so the properties
the conclusions rest on are asserted here rather than taken on trust:

- the simulator is reproducible, heterogeneous, and non-stationary — if it were
  not, the claim that context and continual learning matter would be measuring
  an artefact;
- the runner honours the safety gate, computes regret against the oracle
  restricted to the *same* gated set, and logs propensities at decision time.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.policies import (
    EpsilonGreedyPolicy,
    LinTSPolicy,
    RandomPolicy,
    RulePolicy,
    build_policy,
)
from src.evaluation.runner import (
    collect_logged_data,
    compute_metrics,
    convergence_episode,
    rolling_mean,
    run_policy,
)
from src.feature_store.transform import FEATURE_DIM
from src.recommendation.action_space import ActionSpace
from src.simulation.fitness_env import FitnessSimulator

N_ACTIONS = 18


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_simulator_is_reproducible():
    """Same seed, same trajectory — every reported number depends on this."""

    def trajectory():
        sim = FitnessSimulator(seed=42)
        user = sim.new_user()
        out = []
        for _ in range(50):
            state = sim.observe(user)
            out.append((state["readiness_score"], sim.step(user, state, 5)))
        return out

    assert trajectory() == trajectory()


def test_rewards_stay_in_range():
    sim = FitnessSimulator(seed=1)
    space = ActionSpace()
    for _ in range(5):
        user = sim.new_user()
        for _ in range(30):
            state = sim.observe(user)
            for aid in range(N_ACTIONS):
                assert (
                    -1.0
                    <= sim.expected_reward(user, state, space.get_action(aid))
                    <= 1.0
                )
            assert (
                -1.0 <= sim.step(user, state, int(np.random.randint(N_ACTIONS))) <= 1.0
            )


def test_expected_reward_is_noise_free_and_deterministic():
    """
    The ground truth used for regret and for validating off-policy estimators
    must not itself be stochastic.
    """
    sim = FitnessSimulator(seed=2)
    user = sim.new_user()
    state = sim.observe(user)
    action = ActionSpace().get_action(7)
    values = [sim.expected_reward(user, state, action) for _ in range(20)]
    assert len(set(values)) == 1


def test_sampled_reward_is_centred_on_the_expected_reward():
    sim = FitnessSimulator(seed=3)
    user = sim.new_user()
    state = sim.observe(user)
    action = ActionSpace().get_action(9)

    expected = sim.expected_reward(user, state, action)
    samples = [sim.sample_reward(user, state, action) for _ in range(4000)]
    assert np.mean(samples) == pytest.approx(expected, abs=0.02)


def test_users_are_heterogeneous():
    """
    Different users must prefer different actions in comparable states.  If they
    did not, there would be nothing to personalise and a context-free bandit
    would be sufficient.
    """
    sim = FitnessSimulator(seed=4)
    users = [sim.new_user() for _ in range(60)]
    optima = set()
    for user in users:
        state = sim.observe(user)
        optima.add(sim.optimal_action(user, state))
    assert len(optima) >= 5


def test_latent_traits_are_not_fully_revealed_by_the_profile():
    """
    Stated tolerance is a noisy view of the latent value.  If it were exact, the
    problem would be fully observable and the bandit would face no real
    uncertainty.
    """
    sim = FitnessSimulator(seed=5)
    users = [sim.new_user() for _ in range(200)]
    stated = np.array([u.profile.stated_intensity_tolerance for u in users])
    latent = np.array([u.profile.latent_intensity_tolerance for u in users])
    corr = float(np.corrcoef(stated, latent)[0, 1])
    assert 0.4 < corr < 0.98


def test_fitness_adapts_to_training_load():
    """Non-stationarity: consistent load builds fitness, rest slowly erodes it."""
    sim = FitnessSimulator(seed=6)

    trained = sim.new_user()
    start_trained = trained.fitness_level
    for _ in range(120):
        trained_state = sim.observe(trained)
        sim.step(trained, trained_state, 6)  # STRENGTH MEDIUM 45min

    rested = sim.new_user()
    start_rested = rested.fitness_level
    for _ in range(120):
        sim.step(rested, sim.observe(rested), 0)  # REST

    assert trained.fitness_level > start_trained
    assert rested.fitness_level < start_rested


def test_detraining_is_slower_than_adaptation():
    """
    The asymmetry that stops a myopic policy from spiralling into permanent
    rest. Equal-length training and rest blocks must not cancel out.
    """
    sim = FitnessSimulator(seed=7)
    user = sim.new_user()
    baseline = user.fitness_level

    for _ in range(60):
        sim.step(user, sim.observe(user), 6)
    peak = user.fitness_level
    for _ in range(60):
        sim.step(user, sim.observe(user), 0)
    after_rest = user.fitness_level

    assert peak > baseline
    assert after_rest > baseline  # rest undid less than training built


def test_optimal_action_respects_the_allowed_set():
    sim = FitnessSimulator(seed=8)
    user = sim.new_user()
    state = sim.observe(user)
    allowed = [0, 3, 11]
    assert sim.optimal_action(user, state, allowed) in allowed


def test_optimal_value_is_the_maximum_over_allowed_actions():
    sim = FitnessSimulator(seed=9)
    space = ActionSpace()
    user = sim.new_user()
    state = sim.observe(user)
    allowed = [2, 5, 9, 14]

    best = sim.optimal_value(user, state, allowed)
    for aid in allowed:
        assert sim.expected_reward(user, state, space.get_action(aid)) <= best + 1e-12


def test_adaptation_can_be_disabled():
    sim = FitnessSimulator(seed=10, enable_adaptation=False)
    user = sim.new_user()
    start = user.fitness_level
    for _ in range(80):
        sim.step(user, sim.observe(user), 6)
    assert user.fitness_level == start


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["random", "rule_based", "beta_ts", "lin_ts", "neural_linear"]
)
def test_every_policy_runs_and_honours_the_gate(key):
    policy = build_policy(key, N_ACTIONS, FEATURE_DIM, seed=0)
    records = run_policy(policy, n_episodes=120, n_users=12, seed=0)

    assert len(records) == 120
    for r in records:
        assert r.action in r.allowed
        assert r.features.shape == (FEATURE_DIM,)


def test_regret_is_non_negative_and_bounded_by_the_oracle():
    policy = build_policy("lin_ts", N_ACTIONS, FEATURE_DIM, seed=1)
    records = run_policy(policy, n_episodes=200, n_users=20, seed=1)
    for r in records:
        assert r.regret >= 0.0
        assert r.oracle_value >= r.expected_reward - 1e-12


def test_optimal_flag_implies_zero_regret():
    policy = build_policy("lin_ts", N_ACTIONS, FEATURE_DIM, seed=2)
    for r in run_policy(policy, n_episodes=200, n_users=20, seed=2):
        if r.is_optimal:
            assert r.regret == pytest.approx(0.0, abs=1e-12)


def test_rule_policy_covers_far_less_of_the_action_space_than_a_bandit():
    """
    The quantified version of "a deterministic rule set can never surface most
    of the catalogue".
    """
    rule = compute_metrics(
        run_policy(
            build_policy("rule_based", N_ACTIONS, FEATURE_DIM),
            n_episodes=400,
            n_users=40,
            seed=3,
        )
    )
    bandit = compute_metrics(
        run_policy(
            build_policy("lin_ts", N_ACTIONS, FEATURE_DIM, seed=3),
            n_episodes=400,
            n_users=40,
            seed=3,
        )
    )
    assert rule["actions_used"] <= 5
    assert bandit["actions_used"] >= 15


def test_compute_metrics_reports_the_expected_fields():
    records = run_policy(
        build_policy("random", N_ACTIONS, FEATURE_DIM, seed=4),
        n_episodes=150,
        n_users=15,
        seed=4,
    )
    m = compute_metrics(records)
    for key in (
        "mean_reward",
        "cumulative_regret",
        "final_regret_rate",
        "optimal_action_rate",
        "actions_used",
        "mean_auc",
        "regret_curve",
    ):
        assert key in m
    assert m["n_episodes"] == 150
    assert len(m["regret_curve"]) == 150
    assert m["cumulative_regret"] == pytest.approx(m["regret_curve"][-1])


def test_compute_metrics_rejects_an_empty_run():
    with pytest.raises(ValueError):
        compute_metrics([])


def test_rolling_mean_matches_its_definition():
    values = [1.0, 2.0, 3.0, 4.0]
    assert rolling_mean(values, window=2) == [1.0, 1.5, 2.5, 3.5]


def test_convergence_episode_is_within_the_run():
    records = run_policy(
        build_policy("lin_ts", N_ACTIONS, FEATURE_DIM, seed=5),
        n_episodes=300,
        n_users=30,
        seed=5,
    )
    assert 0 <= convergence_episode(records) <= len(records)


def test_random_policy_beats_nothing_but_stays_inside_the_gate():
    """A uniform policy still must never emit an action the gate excluded."""
    records = run_policy(
        RandomPolicy(N_ACTIONS, seed=6), n_episodes=300, n_users=30, seed=6
    )
    assert all(r.action in r.allowed for r in records)


# ---------------------------------------------------------------------------
# Logged-data collection
# ---------------------------------------------------------------------------


def test_collect_logged_data_produces_a_usable_batch():
    behaviour = EpsilonGreedyPolicy(RulePolicy(N_ACTIONS), epsilon=0.5, seed=7)
    batch = collect_logged_data(behaviour, n_episodes=300, n_users=30, seed=7)

    assert batch.n == 300
    assert batch.contexts.shape == (300, FEATURE_DIM)
    assert np.all(batch.propensities > 0.0)
    assert batch.true_q.shape == (300, N_ACTIONS)
    assert np.all(np.isfinite(batch.true_q))


def test_logged_actions_are_always_inside_the_gate():
    behaviour = EpsilonGreedyPolicy(RulePolicy(N_ACTIONS), epsilon=0.5, seed=8)
    batch = collect_logged_data(behaviour, n_episodes=300, n_users=30, seed=8)
    for i in range(batch.n):
        assert batch.allowed_masks[i, batch.actions[i]]


def test_epsilon_greedy_gives_every_gated_action_positive_probability():
    """
    The identification condition for inverse-propensity estimation.  An action
    with zero logging probability is one the log can say nothing about.
    """
    behaviour = EpsilonGreedyPolicy(RulePolicy(N_ACTIONS), epsilon=0.4, seed=9)
    batch = collect_logged_data(behaviour, n_episodes=200, n_users=20, seed=9)

    for i in range(batch.n):
        allowed = np.flatnonzero(batch.allowed_masks[i])
        probs = behaviour.propensities(batch.contexts[i], allowed)
        assert np.all(probs[allowed] > 0.0)
        assert probs.sum() == pytest.approx(1.0)


def test_epsilon_greedy_rejects_an_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyPolicy(RulePolicy(N_ACTIONS), epsilon=0.0)


def test_propensities_batch_matches_the_row_wise_version_for_fixed_policies():
    behaviour = EpsilonGreedyPolicy(RulePolicy(N_ACTIONS), epsilon=0.3, seed=10)
    batch = collect_logged_data(behaviour, n_episodes=100, n_users=10, seed=10)

    batched = behaviour.propensities_batch(batch.contexts, batch.allowed_masks)
    for i in range(batch.n):
        allowed = np.flatnonzero(batch.allowed_masks[i])
        assert np.allclose(
            batched[i], behaviour.propensities(batch.contexts[i], allowed)
        )


def test_logging_policy_does_not_learn_by_default():
    """
    A fixed logging policy keeps the logged data i.i.d. given the context
    stream, which is what the estimators assume.  ``learn=True`` is opt-in.
    """
    base = LinTSPolicy(N_ACTIONS, FEATURE_DIM, seed=11)
    behaviour = EpsilonGreedyPolicy(base, epsilon=0.5, seed=11)
    collect_logged_data(behaviour, n_episodes=200, n_users=20, seed=11)
    assert base.model.statistics()["n_observations"] == 0
