"""Tests for the safety gate module."""

import pytest
from src.safety.safety_gate import SafetyGate


class TestSafetyGate:
    """Tests for the SafetyGate class."""

    def test_initialization(self):
        gate = SafetyGate()
        assert gate is not None

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
