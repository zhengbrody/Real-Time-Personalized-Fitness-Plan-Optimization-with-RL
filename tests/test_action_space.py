"""Tests for the action space module."""

import pytest
from src.recommendation.action_space import ActionSpace, Action


class TestAction:
    """Tests for the Action dataclass."""

    def test_action_creation(self):
        action = Action(
            action_id=0,
            workout_type="CARDIO",
            intensity="MEDIUM",
            duration_minutes=30,
            description="A moderate jog",
        )
        assert action.action_id == 0
        assert action.workout_type == "CARDIO"
        assert action.intensity == "MEDIUM"
        assert action.duration_minutes == 30

    def test_action_str(self):
        action = Action(
            action_id=1,
            workout_type="REST",
            intensity="NONE",
            duration_minutes=0,
            description="Full rest",
        )
        result = str(action)
        assert "REST" in result or "Rest" in result


class TestActionSpace:
    """Tests for the ActionSpace class."""

    def test_default_actions_exist(self):
        space = ActionSpace()
        actions = space.get_all_actions()
        assert len(actions) > 0

    def test_get_action_by_index(self):
        space = ActionSpace()
        action = space.get_action(0)
        assert isinstance(action, Action)

    def test_action_count(self):
        space = ActionSpace()
        count = space.get_action_count()
        assert count > 0
        assert count == len(space.get_all_actions())

    def test_filter_by_safety(self):
        space = ActionSpace()
        rest_actions = space.filter_by_safety(allowed_types=["REST"])
        for action_id in rest_actions:
            action = space.get_action(action_id)
            assert action.workout_type == "REST"

    def test_filter_by_max_intensity(self):
        space = ActionSpace()
        low_actions = space.filter_by_safety(
            allowed_types=["REST", "RECOVERY", "STRENGTH", "CARDIO"],
            max_intensity="LOW",
        )
        for action_id in low_actions:
            action = space.get_action(action_id)
            assert action.intensity in ["NONE", "LOW"]
