"""Tests for the reward function module."""

import pytest
from src.recommendation.reward_fn import RewardFunction


class TestRewardFunction:
    """Tests for the RewardFunction class."""

    def test_initialization(self):
        rf = RewardFunction()
        assert hasattr(rf, "compute_reward")
        assert callable(rf.compute_reward)

    def test_positive_reward_for_completion(self):
        rf = RewardFunction()
        reward = rf.compute_reward(
            completion=1.0,
            adherence_ratio=1.0,
            recovery_change=0.1,
            satisfaction=0.8,
            overtraining=False,
        )
        assert reward > 0

    def test_negative_reward_for_overtraining(self):
        rf = RewardFunction()
        reward = rf.compute_reward(
            completion=1.0,
            adherence_ratio=1.0,
            recovery_change=-0.2,
            satisfaction=0.3,
            overtraining=True,
        )
        # Overtraining penalty should pull reward down significantly
        assert reward < 1.0

    def test_zero_completion_low_reward(self):
        rf = RewardFunction()
        reward = rf.compute_reward(
            completion=0.0,
            adherence_ratio=0.0,
            recovery_change=0.0,
            satisfaction=0.5,
            overtraining=False,
        )
        assert reward < 1.0

    def test_reward_is_numeric(self):
        rf = RewardFunction()
        reward = rf.compute_reward(
            completion=1.0,
            adherence_ratio=0.5,
            recovery_change=0.0,
            satisfaction=0.5,
            overtraining=False,
        )
        assert isinstance(reward, (int, float))

    def test_compute_reward_from_dict(self):
        rf = RewardFunction()
        outcomes = {
            "completion": 1.0,
            "adherence_ratio": 0.8,
            "recovery_change": 0.1,
            "satisfaction": 0.7,
            "overtraining": False,
        }
        reward = rf.compute_reward_from_dict(outcomes)
        assert isinstance(reward, (int, float))
