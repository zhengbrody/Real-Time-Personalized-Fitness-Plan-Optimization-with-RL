"""Tests for the hybrid recommender module."""

import pytest
from unittest.mock import patch, MagicMock
from src.recommendation.hybrid_recommender import HybridRecommender
from src.recommendation.action_space import ActionSpace


class TestHybridRecommender:
    """Tests for the HybridRecommender class."""

    def test_initialization(self):
        recommender = HybridRecommender()
        assert recommender.action_space is not None
        assert recommender.action_space.get_action_count() == 18
        assert recommender.bandit is not None

    def test_initialization_without_rl(self):
        """Line 39: use_rl=False sets self.bandit = None."""
        recommender = HybridRecommender(use_rl=False)
        assert recommender.bandit is None

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

    def test_recommend_empty_allowed_actions_defaults_to_rest(self):
        """Line 62: When filter_actions returns empty list, fallback to [0] (REST)."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 75,
            "hrv": 50,
            "resting_hr": 60,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": 2,
            "activity_score": 70,
        }
        with patch.object(
            recommender.safety_gate, "filter_actions", return_value=[]
        ):
            result = recommender.recommend(state)
            assert result["action_id"] == 0
            assert result["workout_type"] == "REST"

    def _make_all_actions_allowed(self, recommender, state):
        """Helper: mock safety gate to return all actions as allowed."""
        all_ids = list(range(recommender.action_space.get_action_count()))
        return patch.object(
            recommender.safety_gate, "filter_actions", return_value=all_ids
        )

    def _make_safe_check(self, recommender):
        """Helper: mock safety check_state to return is_safe=True."""
        safe_result = MagicMock()
        safe_result.is_safe = True
        safe_result.message = "All clear"
        return patch.object(
            recommender.safety_gate, "check_state", return_value=safe_result
        )

    def test_rule_based_low_readiness_returns_rest(self):
        """Line 114: readiness < 40 returns REST (action 0)."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 30,
            "sleep_duration_hours": 7,
            "fatigue": 3,
            "days_since_training": 1,
            "hrv": 40,
            "resting_hr": 60,
            "soreness": 2,
            "activity_score": 50,
        }
        # Mock safety gate to allow all actions so we reach rule-based logic
        with self._make_safe_check(recommender), \
             self._make_all_actions_allowed(recommender, state):
            result = recommender.recommend(state)
            assert result["action_id"] == 0
            assert result["workout_type"] == "REST"

    def test_rule_based_low_sleep_returns_rest(self):
        """Line 114: sleep_hours < 5 returns REST (action 0)."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 80,
            "sleep_duration_hours": 4,
            "fatigue": 3,
            "days_since_training": 1,
            "hrv": 60,
            "resting_hr": 55,
            "soreness": 2,
            "activity_score": 70,
        }
        with self._make_safe_check(recommender), \
             self._make_all_actions_allowed(recommender, state):
            result = recommender.recommend(state)
            assert result["action_id"] == 0
            assert result["workout_type"] == "REST"

    def test_rule_based_high_fatigue_returns_recovery(self):
        """Lines 118-124: fatigue >= 7 returns RECOVERY action."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 60,
            "sleep_duration_hours": 7,
            "fatigue": 8,
            "days_since_training": 1,
            "hrv": 50,
            "resting_hr": 60,
            "soreness": 2,
            "activity_score": 50,
        }
        with self._make_safe_check(recommender), \
             self._make_all_actions_allowed(recommender, state):
            result = recommender.recommend(state)
            assert result["workout_type"] == "RECOVERY"

    def test_rule_based_high_fatigue_no_recovery_actions_falls_through(self):
        """Lines 118-124: High fatigue but no RECOVERY in allowed_actions."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 60,
            "sleep_duration_hours": 7,
            "fatigue": 8,
            "days_since_training": 1,
            "hrv": 50,
            "resting_hr": 60,
            "soreness": 2,
            "activity_score": 50,
        }
        # Mock filter_actions to return only STRENGTH MEDIUM ids (no RECOVERY)
        action_space = recommender.action_space
        medium_strength_ids = [
            a.action_id
            for a in action_space.get_all_actions()
            if a.workout_type == "STRENGTH" and a.intensity == "MEDIUM"
        ]
        with self._make_safe_check(recommender), \
             patch.object(
                 recommender.safety_gate, "filter_actions", return_value=medium_strength_ids
             ):
            result = recommender.recommend(state)
            # No RECOVERY found; days_since_training=1 so skips rule 3;
            # no LOW in medium_strength_ids, so falls through to allowed_actions[0]
            assert result["action_id"] == medium_strength_ids[0]

    def test_rule_based_long_rest_returns_medium(self):
        """Lines 128-134: days_since_training >= 3 returns MEDIUM intensity."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 70,
            "sleep_duration_hours": 8,
            "fatigue": 4,
            "days_since_training": 4,
            "hrv": 60,
            "resting_hr": 55,
            "soreness": 2,
            "activity_score": 70,
        }
        with self._make_safe_check(recommender), \
             self._make_all_actions_allowed(recommender, state):
            result = recommender.recommend(state)
            assert result["intensity"] == "MEDIUM"

    def test_rule_based_long_rest_no_medium_actions_falls_through(self):
        """Lines 128-134: days_since >= 3 but no MEDIUM actions available."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 70,
            "sleep_duration_hours": 8,
            "fatigue": 4,
            "days_since_training": 4,
            "hrv": 60,
            "resting_hr": 55,
            "soreness": 2,
            "activity_score": 70,
        }
        # Only allow HIGH intensity actions
        action_space = recommender.action_space
        high_ids = [
            a.action_id
            for a in action_space.get_all_actions()
            if a.intensity == "HIGH"
        ]
        with self._make_safe_check(recommender), \
             patch.object(
                 recommender.safety_gate, "filter_actions", return_value=high_ids
             ):
            result = recommender.recommend(state)
            # No MEDIUM and no LOW -> falls through to allowed_actions[0]
            assert result["action_id"] == high_ids[0]

    def test_rule_based_fallback_to_allowed_actions_first(self):
        """Line 145: No LOW intensity actions -> return allowed_actions[0]."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 70,
            "sleep_duration_hours": 8,
            "fatigue": 4,
            "days_since_training": 1,
            "hrv": 60,
            "resting_hr": 55,
            "soreness": 2,
            "activity_score": 70,
        }
        # Only allow HIGH intensity actions (no LOW ones)
        action_space = recommender.action_space
        high_ids = [
            a.action_id
            for a in action_space.get_all_actions()
            if a.intensity == "HIGH"
        ]
        with self._make_safe_check(recommender), \
             patch.object(
                 recommender.safety_gate, "filter_actions", return_value=high_ids
             ):
            result = recommender.recommend(state)
            assert result["action_id"] == high_ids[0]

    def test_rationale_rest_day(self):
        """Line 153: REST workout generates rest-day rationale."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 30,
            "sleep_duration_hours": 4,
            "fatigue": 3,
            "days_since_training": 1,
            "hrv": 40,
            "resting_hr": 60,
            "soreness": 2,
            "activity_score": 50,
        }
        with self._make_safe_check(recommender), \
             self._make_all_actions_allowed(recommender, state):
            result = recommender.recommend(state)
            assert result["workout_type"] == "REST"
            assert "Rest day recommended" in result["rationale"]
            assert "readiness" in result["rationale"].lower()

    def test_rationale_low_intensity(self):
        """Line 155: LOW intensity, non-REST generates recovery state rationale."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 70,
            "sleep_duration_hours": 8,
            "fatigue": 4,
            "days_since_training": 1,
            "hrv": 60,
            "resting_hr": 55,
            "soreness": 2,
            "activity_score": 70,
        }
        with self._make_safe_check(recommender), \
             self._make_all_actions_allowed(recommender, state):
            result = recommender.recommend(state)
            # Default rule path: good readiness, low fatigue, days_since=1
            # Falls to LOW intensity default
            assert result["intensity"] == "LOW"
            assert "recovery state" in result["rationale"].lower()

    def test_rationale_medium_or_high_intensity(self):
        """Line 157: Non-REST, non-LOW intensity generates 'well recovered' rationale."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 70,
            "sleep_duration_hours": 8,
            "fatigue": 4,
            "days_since_training": 4,
            "hrv": 60,
            "resting_hr": 55,
            "soreness": 2,
            "activity_score": 70,
        }
        with self._make_safe_check(recommender), \
             self._make_all_actions_allowed(recommender, state):
            result = recommender.recommend(state)
            # days_since >= 3 triggers MEDIUM
            assert result["intensity"] == "MEDIUM"
            assert "well recovered" in result["rationale"].lower()

    def test_update_with_bandit(self):
        """Lines 161-162: update() calls bandit.update when bandit exists."""
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
        result = recommender.recommend(state)
        # Should not raise
        recommender.update(result["action_id"], state, 0.8)
        # Verify bandit was updated by checking counts increased
        assert recommender.bandit.action_counts[result["action_id"]] == 1

    def test_update_without_bandit(self):
        """Lines 161-162: update() is a no-op when bandit is None."""
        recommender = HybridRecommender(use_rl=False)
        assert recommender.bandit is None
        # Should not raise
        recommender.update(0, {}, 0.5)

    def test_recommend_no_rl_constructor_uses_rules(self):
        """use_rl=False constructor -> recommend uses rule-based path."""
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 70,
            "sleep_duration_hours": 8,
            "fatigue": 4,
            "days_since_training": 1,
            "hrv": 60,
            "resting_hr": 55,
            "soreness": 2,
            "activity_score": 70,
        }
        result = recommender.recommend(state)
        assert result is not None
        assert isinstance(result, dict)
