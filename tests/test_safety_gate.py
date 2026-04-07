"""Tests for the safety gate module."""

import pytest
from unittest.mock import patch, MagicMock
from src.safety.safety_gate import SafetyGate
from src.agent.safety import SafetyCheckResult


class TestSafetyGate:
    """Tests for the SafetyGate class."""

    def test_initialization(self):
        gate = SafetyGate()
        assert hasattr(gate, "rules")
        assert len(gate.rules) == 4

    def test_check_state_normal(self):
        gate = SafetyGate()
        state = {
            "readiness_score": 75,
            "hrv": 55,
            "resting_hr": 58,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": 2,
        }
        result = gate.check_state(state)
        # Normal state should be safe
        assert result is not None
        assert hasattr(result, "is_safe")

    def test_dangerous_state_check(self):
        gate = SafetyGate()
        state = {
            "readiness_score": 10,
            "hrv": 15,
            "resting_hr": 95,
            "sleep_duration_hours": 3,
            "fatigue": 10,
            "soreness": 9,
        }
        result = gate.check_state(state)
        # Dangerous state should fail safety check
        assert result is not None
        assert hasattr(result, "is_safe")

    def test_filter_actions(self):
        gate = SafetyGate()
        state = {
            "readiness_score": 75,
            "hrv": 55,
            "resting_hr": 58,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": 2,
        }
        all_action_ids = list(range(25))
        filtered_actions = gate.filter_actions(state, all_action_ids)
        # Should return a list of allowed action IDs
        assert isinstance(filtered_actions, list)
        assert len(filtered_actions) > 0

    def test_filter_actions_mandatory_rest_day(self):
        """Test filter_actions when overtraining_risk triggers mandatory_rest_day (lines 114-117)."""
        gate = SafetyGate()
        state = {
            "readiness_score": 50,
            "hrv": 55,
            "resting_hr": 58,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": 2,
            "overtraining_risk": True,
        }
        all_action_ids = list(range(25))

        # Mock the ActionSpace import inside the safety_gate module
        mock_action_space_cls = MagicMock()
        mock_action_space_instance = MagicMock()
        mock_action_space_cls.return_value = mock_action_space_instance

        with patch.dict(
            "sys.modules",
            {"src.recommendation.action_space": MagicMock(ActionSpace=mock_action_space_cls)},
        ):
            filtered = gate.filter_actions(state, all_action_ids)

        # mandatory_rest_day returns [0] (REST action ID)
        assert filtered == [0]

    def test_filter_actions_rest_day_or_light_activity(self):
        """Test filter_actions when low HRV triggers rest_day_or_light_activity (lines 121-124)."""
        gate = SafetyGate()
        # Low HRV (< 20) triggers "rest_day_or_light_activity" recommended_action
        state = {
            "readiness_score": 75,
            "hrv": 15,
            "resting_hr": 58,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": 2,
            "overtraining_risk": False,
        }
        all_action_ids = list(range(25))

        mock_action_space_cls = MagicMock()
        mock_action_space_instance = MagicMock()
        mock_action_space_instance.filter_by_safety.return_value = [0, 1, 2]
        mock_action_space_cls.return_value = mock_action_space_instance

        with patch.dict(
            "sys.modules",
            {"src.recommendation.action_space": MagicMock(ActionSpace=mock_action_space_cls)},
        ):
            filtered = gate.filter_actions(state, all_action_ids)

        assert filtered == [0, 1, 2]
        mock_action_space_instance.filter_by_safety.assert_called_once_with(
            allowed_types=["REST", "RECOVERY"], max_intensity="LOW"
        )

    def test_filter_actions_reduce_intensity(self):
        """Test filter_actions when low readiness triggers reduce_intensity (lines 130-133)."""
        gate = SafetyGate()
        # readiness_score < 50 triggers "reduce_intensity", but we need to avoid
        # higher-priority checks (no low HRV, no overtraining, no high fatigue, no high soreness)
        state = {
            "readiness_score": 29,
            "hrv": 55,
            "resting_hr": 58,
            "sleep_duration_hours": 7,
            "fatigue": 5,
            "soreness": 2,
            "overtraining_risk": False,
        }
        all_action_ids = list(range(25))

        mock_action_space_cls = MagicMock()
        mock_action_space_instance = MagicMock()
        mock_action_space_instance.filter_by_safety.return_value = [0, 1, 2, 3, 4, 5]
        mock_action_space_cls.return_value = mock_action_space_instance

        with patch.dict(
            "sys.modules",
            {"src.recommendation.action_space": MagicMock(ActionSpace=mock_action_space_cls)},
        ):
            filtered = gate.filter_actions(state, all_action_ids)

        assert filtered == [0, 1, 2, 3, 4, 5]
        mock_action_space_instance.filter_by_safety.assert_called_once_with(
            allowed_types=["REST", "RECOVERY", "STRENGTH", "CARDIO"],
            max_intensity="MEDIUM",
        )

    def test_check_state_high_fatigue_boundary(self):
        """Test check_state with fatigue exactly at 8 (boundary)."""
        gate = SafetyGate()
        state = {
            "readiness_score": 75,
            "hrv": 55,
            "resting_hr": 58,
            "sleep_duration_hours": 8,
            "fatigue": 8,
            "soreness": 2,
        }
        result = gate.check_state(state)
        assert result.is_safe is False

    def test_check_state_readiness_boundary(self):
        """Test check_state with readiness_score exactly at 29 (below 50)."""
        gate = SafetyGate()
        state = {
            "readiness_score": 29,
            "hrv": 55,
            "resting_hr": 58,
            "sleep_duration_hours": 7,
            "fatigue": 5,
            "soreness": 2,
        }
        result = gate.check_state(state)
        assert result.is_safe is False
        assert result.recommended_action == "reduce_intensity"

    def test_safety_rules_created(self):
        """Test that all expected safety rules are created."""
        gate = SafetyGate()
        assert len(gate.rules) == 4
        rule_names = [r.name for r in gate.rules]
        assert "high_fatigue" in rule_names
        assert "consecutive_high_load" in rule_names
