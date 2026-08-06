"""Tests for the contextual bandits module."""

import numpy as np
import pytest
from src.recommendation.contextual_bandits import (
    ContextualBandit,
    LinearContextualBandit,
)
from src.recommendation.action_space import ActionSpace


class TestContextualBandit:
    """Tests for the basic ContextualBandit (Thompson Sampling)."""

    def test_initialization(self):
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        assert bandit.action_space is not None
        assert len(bandit.alpha) == action_space.get_action_count()

    def test_select_action(self):
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        context = np.zeros(1)
        action = bandit.select_action(context)
        assert 0 <= action < action_space.get_action_count()

    def test_update(self):
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        initial_alpha = bandit.alpha[0]
        # Update action 0 with positive reward (>0.5)
        bandit.update(action_id=0, reward=1.0)
        # Alpha should increase for action 0
        assert bandit.alpha[0] > initial_alpha

    def test_update_negative_reward(self):
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        initial_beta = bandit.beta[1]
        bandit.update(action_id=1, reward=0.0)
        # Beta should increase for action 1
        assert bandit.beta[1] > initial_beta

    def test_repeated_updates_bias_selection(self):
        """After many positive updates, action should be selected more often."""
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        context = np.zeros(1)
        # Give action 0 many positive rewards
        for _ in range(100):
            bandit.update(action_id=0, reward=1.0)

        # Sample many times and check action 0 is preferred
        selections = [bandit.select_action(context) for _ in range(200)]
        action_0_count = selections.count(0)
        assert action_0_count > 50  # Should be selected more often

    def test_get_action_probabilities_default_allowed(self):
        """Test get_action_probabilities with default (all) allowed_actions (lines 104-122)."""
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        context = np.zeros(1)

        probs = bandit.get_action_probabilities(context)
        # Should return a dict for all actions
        assert isinstance(probs, dict)
        assert len(probs) == action_space.get_action_count()
        # Probabilities should sum to approximately 1.0
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        # All values should be non-negative
        assert all(v >= 0 for v in probs.values())

    def test_get_action_probabilities_with_allowed_actions(self):
        """Test get_action_probabilities with a restricted set of allowed_actions (lines 104-122)."""
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        context = np.zeros(1)

        allowed = [0, 1, 2]
        probs = bandit.get_action_probabilities(context, allowed_actions=allowed)
        # Should only return probabilities for allowed actions
        assert set(probs.keys()) == set(allowed)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_get_action_probabilities_after_updates(self):
        """Test that probabilities shift after updates."""
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        context = np.zeros(1)

        # Before updates, all should be roughly equal (uniform prior)
        probs_before = bandit.get_action_probabilities(context)

        # Give action 0 many positive updates
        for _ in range(50):
            bandit.update(action_id=0, reward=1.0)

        probs_after = bandit.get_action_probabilities(context)
        # Action 0 should have a higher probability now
        assert probs_after[0] > probs_before[0]

    def test_get_action_probabilities_single_action(self):
        """Test get_action_probabilities with a single allowed action."""
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        context = np.zeros(1)

        probs = bandit.get_action_probabilities(context, allowed_actions=[3])
        assert probs == {3: 1.0}

    def test_get_statistics(self):
        """Test get_statistics returns correct structure (line 126)."""
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)

        # Do some updates first
        bandit.update(action_id=0, reward=1.0)
        bandit.update(action_id=0, reward=0.8)
        bandit.update(action_id=1, reward=0.2)

        stats = bandit.get_statistics()
        assert "action_counts" in stats
        assert "total_rewards" in stats
        assert "expected_rewards" in stats
        # action 0 should have count 2
        assert stats["action_counts"][0] == 2
        # action 1 should have count 1
        assert stats["action_counts"][1] == 1
        # total_rewards for action 0 should be 1.8
        assert abs(stats["total_rewards"][0] - 1.8) < 1e-6
        # expected_rewards should be a dict
        assert isinstance(stats["expected_rewards"], dict)

    def test_get_statistics_initial(self):
        """Test get_statistics with no updates."""
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)

        stats = bandit.get_statistics()
        # All counts should be zero
        assert all(c == 0 for c in stats["action_counts"])
        # All expected rewards should be 0.5 (uniform Beta(1,1))
        for v in stats["expected_rewards"].values():
            assert abs(v - 0.5) < 1e-6

    def test_get_action_probabilities_zero_total(self):
        """Test get_action_probabilities when all expected rewards are zero (line 120)."""
        action_space = ActionSpace()
        bandit = ContextualBandit(action_space=action_space)
        context = np.zeros(1)

        # Force alpha to zero for specific actions so expected = 0/(0+beta) = 0
        allowed = [0, 1, 2]
        for a in allowed:
            bandit.alpha[a] = 0.0

        probs = bandit.get_action_probabilities(context, allowed_actions=allowed)
        # Should fall back to uniform distribution
        expected_prob = 1.0 / len(allowed)
        for a in allowed:
            assert abs(probs[a] - expected_prob) < 1e-6


class TestLinearContextualBandit:
    """Tests for the LinearContextualBandit."""

    def test_initialization(self):
        action_space = ActionSpace()
        bandit = LinearContextualBandit(action_space=action_space, feature_dim=7)
        assert bandit.action_space is not None
        assert bandit.feature_dim == 7

    def test_select_action_with_context(self):
        action_space = ActionSpace()
        bandit = LinearContextualBandit(action_space=action_space, feature_dim=7)
        rng = np.random.default_rng(42)
        context = rng.standard_normal(7)
        action = bandit.select_action(context)
        assert 0 <= action < action_space.get_action_count()

    def test_update_with_context(self):
        action_space = ActionSpace()
        bandit = LinearContextualBandit(action_space=action_space, feature_dim=4)
        rng = np.random.default_rng(42)
        context = rng.standard_normal(4)
        bandit.update(action_id=0, context=context, reward=1.0)
        # Should not raise any error

    def test_get_expected_reward(self):
        """Test get_expected_reward returns a float (lines 230-231)."""
        action_space = ActionSpace()
        feature_dim = 4
        bandit = LinearContextualBandit(
            action_space=action_space, feature_dim=feature_dim
        )
        rng = np.random.default_rng(42)
        context = rng.standard_normal(feature_dim)

        # Before any updates, expected reward should be 0 (theta initialized to zeros)
        reward = bandit.get_expected_reward(action_id=0, context=context)
        assert isinstance(reward, float)
        assert abs(reward) < 1e-6  # Should be ~0 with zero-initialized theta

    def test_get_expected_reward_after_updates(self):
        """Test get_expected_reward changes after updates (lines 230-231)."""
        action_space = ActionSpace()
        feature_dim = 3
        bandit = LinearContextualBandit(
            action_space=action_space, feature_dim=feature_dim
        )

        context = np.array([1.0, 0.5, -0.5])

        # Update several times with positive reward for action 0
        for _ in range(10):
            bandit.update(action_id=0, context=context, reward=1.0)

        # Expected reward should now be non-zero
        reward = bandit.get_expected_reward(action_id=0, context=context)
        assert isinstance(reward, (float, np.floating))
        # With positive rewards and consistent context, expected reward should be positive
        assert reward > 0

    def test_select_action_with_allowed_actions(self):
        """Test select_action with restricted allowed_actions."""
        action_space = ActionSpace()
        bandit = LinearContextualBandit(action_space=action_space, feature_dim=4)
        rng = np.random.default_rng(42)
        context = rng.standard_normal(4)

        allowed = [0, 1, 2]
        action = bandit.select_action(context, allowed_actions=allowed)
        assert action in allowed

    def test_update_modifies_parameters(self):
        """Test that update actually modifies B and f matrices."""
        action_space = ActionSpace()
        feature_dim = 3
        bandit = LinearContextualBandit(
            action_space=action_space, feature_dim=feature_dim
        )

        B_before = bandit.B[0].copy()
        f_before = bandit.f[0].copy()

        context = np.array([1.0, 2.0, 3.0])
        bandit.update(action_id=0, context=context, reward=1.0)

        # B and f should have changed
        assert not np.array_equal(bandit.B[0], B_before)
        assert not np.array_equal(bandit.f[0], f_before)
        assert bandit.action_counts[0] == 1
