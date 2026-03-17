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
        context = np.random.randn(7)
        action = bandit.select_action(context)
        assert 0 <= action < action_space.get_action_count()

    def test_update_with_context(self):
        action_space = ActionSpace()
        bandit = LinearContextualBandit(action_space=action_space, feature_dim=4)
        context = np.random.randn(4)
        bandit.update(action_id=0, context=context, reward=1.0)
        # Should not raise any error
