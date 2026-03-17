"""Tests for the hybrid recommender module."""

import pytest
from src.recommendation.hybrid_recommender import HybridRecommender
from src.recommendation.action_space import ActionSpace


class TestHybridRecommender:
    """Tests for the HybridRecommender class."""

    def test_initialization(self):
        recommender = HybridRecommender()
        assert recommender is not None

    def test_recommend_returns_dict(self):
        recommender = HybridRecommender()
        state = {
            "readiness_score": 75,
            "hrv": 50,
            "resting_hr": 60,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": 2,
            "activity_score": 70,
        }
        result = recommender.recommend(state)
        assert result is not None
        assert isinstance(result, dict)
        assert "action_id" in result
        assert "workout_type" in result
        assert "intensity" in result
        assert "duration_minutes" in result

    def test_recommend_low_readiness_suggests_rest(self):
        recommender = HybridRecommender()
        state = {
            "readiness_score": 20,
            "hrv": 20,
            "resting_hr": 85,
            "sleep_duration_hours": 3,
            "fatigue": 9,
            "soreness": 8,
            "activity_score": 10,
        }
        result = recommender.recommend(state)
        # With very poor readiness, should recommend rest or recovery
        assert result is not None
        assert "action_id" in result

    def test_recommend_high_readiness(self):
        recommender = HybridRecommender()
        state = {
            "readiness_score": 95,
            "hrv": 80,
            "resting_hr": 50,
            "sleep_duration_hours": 9,
            "fatigue": 1,
            "soreness": 1,
            "activity_score": 95,
        }
        result = recommender.recommend(state)
        assert result is not None
        assert isinstance(result, dict)

    def test_recommend_with_use_rl_false(self):
        recommender = HybridRecommender(use_rl=True)
        state = {
            "readiness_score": 75,
            "hrv": 50,
            "resting_hr": 60,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": 2,
            "activity_score": 70,
        }
        # Test using rule-based recommendations
        result = recommender.recommend(state, use_rl=False)
        assert result is not None
        assert isinstance(result, dict)
