"""Tests for the Coach Memory and Session Handoff modules."""

import json
import os
import pytest
import tempfile

from src.agent.memory import CoachMemory
from src.agent.session_handoff import session_handoff


class TestCoachMemory:
    """Tests for the CoachMemory class."""

    def _tmp_path(self, tmp_path):
        """Return a temporary file path for memory JSON."""
        return str(tmp_path / "test_memory.json")

    # ----------------------------------------------------------- basics

    def test_create_empty_memory(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        assert mem.data["user_id"] == "u1"
        assert mem.data["sessions"] == []
        assert mem.data["injury_record"] == []
        assert mem.data["cumulative_stats"]["total_sessions"] == 0

    def test_load_save_roundtrip(self, tmp_path):
        path = self._tmp_path(tmp_path)

        # Save
        mem = CoachMemory(user_id="u1", memory_path=path)
        mem.add_session({"summary": "First session", "compliance_rate": 0.8})
        mem.save()

        # Load into a fresh instance
        mem2 = CoachMemory(user_id="u1", memory_path=path)
        mem2.load()

        assert len(mem2.data["sessions"]) == 1
        assert mem2.data["sessions"][0]["summary"] == "First session"
        assert mem2.data["cumulative_stats"]["total_sessions"] == 1

    def test_memory_file_not_found_creates_new(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        mem = CoachMemory(user_id="u1", memory_path=path)
        mem.load()  # should not raise
        assert mem.data["user_id"] == "u1"
        assert mem.data["sessions"] == []

    # --------------------------------------------------------- sessions

    def test_add_session_updates_stats(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))

        mem.add_session({"summary": "S1", "compliance_rate": 0.8})
        mem.add_session({"summary": "S2", "compliance_rate": 0.6})

        stats = mem.data["cumulative_stats"]
        assert stats["total_sessions"] == 2
        assert stats["total_workouts_completed"] == 2  # both >= 0.5
        assert stats["streak_current"] == 2
        assert stats["streak_best"] == 2
        assert abs(stats["avg_compliance"] - 0.7) < 1e-9

    def test_add_session_streak_breaks_on_low_compliance(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))

        mem.add_session({"summary": "good", "compliance_rate": 1.0})
        mem.add_session({"summary": "good", "compliance_rate": 0.9})
        mem.add_session({"summary": "missed", "compliance_rate": 0.1})
        mem.add_session({"summary": "back", "compliance_rate": 0.8})

        stats = mem.data["cumulative_stats"]
        assert stats["streak_current"] == 1
        assert stats["streak_best"] == 2

    def test_get_recent_sessions(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        for i in range(10):
            mem.add_session({"summary": f"S{i}", "compliance_rate": 0.5})

        recent = mem.get_recent_sessions(3)
        assert len(recent) == 3
        assert recent[0]["summary"] == "S7"
        assert recent[2]["summary"] == "S9"

    # --------------------------------------------------------- injuries

    def test_add_and_resolve_injury(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))

        mem.add_injury({"type": "strain", "body_part": "knee", "severity": "moderate"})
        assert len(mem.data["injury_record"]) == 1
        assert mem.data["injury_record"][0]["resolved_date"] is None

        resolved = mem.resolve_injury("knee")
        assert resolved is True
        assert mem.data["injury_record"][0]["resolved_date"] is not None

    def test_get_active_injuries_filters_resolved(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))

        mem.add_injury({"type": "strain", "body_part": "knee", "severity": "moderate"})
        mem.add_injury({"type": "sprain", "body_part": "ankle", "severity": "mild"})
        mem.resolve_injury("knee")

        active = mem.get_active_injuries()
        assert len(active) == 1
        assert active[0]["body_part"] == "ankle"

    def test_resolve_injury_returns_false_when_none_found(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        assert mem.resolve_injury("shoulder") is False

    # ------------------------------------------------------- preferences

    def test_update_preferences_scalar(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        mem.update_preferences(
            {"goal": "muscle_gain", "preferred_time_of_day": "morning"}
        )

        assert mem.data["preferences"]["goal"] == "muscle_gain"
        assert mem.data["preferences"]["preferred_time_of_day"] == "morning"

    def test_update_preferences_list_deduplicates(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        mem.update_preferences({"preferred_workout_types": ["strength", "cardio"]})
        mem.update_preferences({"preferred_workout_types": ["cardio", "yoga"]})

        assert mem.data["preferences"]["preferred_workout_types"] == [
            "strength",
            "cardio",
            "yoga",
        ]

    # ------------------------------------------------- context summary

    def test_get_context_summary_includes_injuries(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        mem.add_injury({"type": "strain", "body_part": "knee", "severity": "moderate"})

        summary = mem.get_context_summary()
        assert "knee" in summary
        assert "moderate" in summary

    def test_get_context_summary_includes_preferences(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        mem.update_preferences({"goal": "fat_loss"})

        summary = mem.get_context_summary()
        assert "fat_loss" in summary

    def test_get_context_summary_no_history(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        summary = mem.get_context_summary()
        assert summary == "No prior history available."

    def test_get_context_summary_includes_stats(self, tmp_path):
        mem = CoachMemory(user_id="u1", memory_path=self._tmp_path(tmp_path))
        mem.add_session({"summary": "Good workout", "compliance_rate": 1.0})

        summary = mem.get_context_summary()
        assert "1 coaching session" in summary
        assert "100%" in summary


class TestSessionHandoff:
    """Tests for the session_handoff function."""

    def _tmp_path(self, tmp_path):
        return str(tmp_path / "handoff_memory.json")

    def test_session_handoff_persists(self, tmp_path):
        path = self._tmp_path(tmp_path)
        mem = CoachMemory(user_id="u1", memory_path=path)

        session_handoff(
            mem,
            recommendations=["Do 3x10 squats"],
            mood="good",
            compliance_rate=0.9,
            plan={"intensity": "moderate"},
            tools_called=[{"tool": "adjust_plan"}],
        )

        # Memory should have been saved to disk
        assert os.path.exists(path)

        # Reload and verify
        mem2 = CoachMemory(user_id="u1", memory_path=path)
        mem2.load()
        assert len(mem2.data["sessions"]) == 1
        assert mem2.data["sessions"][0]["mood"] == "good"
        assert mem2.data["cumulative_stats"]["total_sessions"] == 1

    def test_session_handoff_summary_content(self, tmp_path):
        path = self._tmp_path(tmp_path)
        mem = CoachMemory(user_id="u1", memory_path=path)

        result = session_handoff(
            mem,
            recommendations=["Rest day"],
            mood="tired",
            plan={"intensity": "low"},
        )

        assert "intensity: low" in result["summary"]
        assert "Mood: tired" in result["summary"]
        assert "1 recommendation" in result["summary"]

    def test_session_handoff_defaults(self, tmp_path):
        path = self._tmp_path(tmp_path)
        mem = CoachMemory(user_id="u1", memory_path=path)

        result = session_handoff(mem)

        assert result["summary"] == "Session completed with no notable events."
        assert result["compliance_rate"] == 0.0
        assert mem.data["cumulative_stats"]["total_sessions"] == 1
