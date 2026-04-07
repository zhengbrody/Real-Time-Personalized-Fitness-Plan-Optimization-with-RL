"""
Coach Memory Manager — Cross-session memory persistence for the AI Coach Agent.

Stores structured user data (session summaries, injuries, preferences, stats)
in a local JSON file so the coach remembers history between conversations.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _default_memory_data(user_id: str) -> Dict:
    """Return an empty memory structure for a new user."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "user_id": user_id,
        "created_at": now,
        "updated_at": now,
        "sessions": [],
        "injury_record": [],
        "preferences": {
            "preferred_workout_types": [],
            "avoided_exercises": [],
            "preferred_time_of_day": None,
            "goal": None,
        },
        "cumulative_stats": {
            "total_sessions": 0,
            "avg_compliance": 0.0,
            "total_workouts_completed": 0,
            "streak_current": 0,
            "streak_best": 0,
        },
    }


class CoachMemory:
    """
    Persistent memory store for the AI Coach Agent.

    Loads/saves from a JSON file and provides helpers for session tracking,
    injury management, preference learning, and LLM context injection.
    """

    def __init__(self, user_id: str, memory_path: str = "coach_memory.json"):
        """
        Initialize coach memory.

        Args:
            user_id: User identifier
            memory_path: Path to the JSON persistence file
        """
        self.user_id = user_id
        self.memory_path = memory_path
        self.data: Dict = _default_memory_data(user_id)

    # ------------------------------------------------------------------ IO

    def load(self) -> None:
        """Load memory from JSON file. Creates empty memory if file does not exist."""
        if not os.path.exists(self.memory_path):
            self.data = _default_memory_data(self.user_id)
            return

        with open(self.memory_path, "r") as f:
            self.data = json.load(f)

    def save(self) -> None:
        """Write current memory to JSON file."""
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.memory_path, "w") as f:
            json.dump(self.data, f, indent=2)

    # ----------------------------------------------------------- Sessions

    def add_session(self, session_data: Dict) -> None:
        """
        Append a session summary and update cumulative stats.

        Args:
            session_data: Dict with keys like summary, recommendations_given,
                          feedback_received, mood, compliance_rate.
        """
        session = {
            "session_id": session_data.get("session_id", str(uuid.uuid4())),
            "date": session_data.get("date", datetime.now(timezone.utc).isoformat()),
            "summary": session_data.get("summary", ""),
            "recommendations_given": session_data.get("recommendations_given", []),
            "feedback_received": session_data.get("feedback_received", {}),
            "mood": session_data.get("mood", ""),
            "compliance_rate": session_data.get("compliance_rate", 0.0),
        }

        self.data["sessions"].append(session)
        self._update_cumulative_stats(session)

    def get_recent_sessions(self, n: int = 5) -> List[Dict]:
        """Return the last *n* session summaries."""
        return self.data["sessions"][-n:]

    # ----------------------------------------------------------- Injuries

    def add_injury(self, injury_data: Dict) -> None:
        """
        Record a new injury.

        Args:
            injury_data: Dict with type, body_part, severity, and optional reported_date.
        """
        injury = {
            "type": injury_data.get("type", "unknown"),
            "body_part": injury_data.get("body_part", "unknown"),
            "severity": injury_data.get("severity", "unknown"),
            "reported_date": injury_data.get(
                "reported_date", datetime.now(timezone.utc).isoformat()
            ),
            "resolved_date": injury_data.get("resolved_date", None),
        }
        self.data["injury_record"].append(injury)

    def resolve_injury(self, body_part: str) -> bool:
        """
        Mark the most recent unresolved injury for *body_part* as resolved.

        Returns:
            True if an injury was resolved, False if none found.
        """
        for injury in reversed(self.data["injury_record"]):
            if injury["body_part"] == body_part and injury["resolved_date"] is None:
                injury["resolved_date"] = datetime.now(timezone.utc).isoformat()
                return True
        return False

    def get_active_injuries(self) -> List[Dict]:
        """Return injuries where resolved_date is None."""
        return [
            i for i in self.data["injury_record"] if i["resolved_date"] is None
        ]

    # -------------------------------------------------------- Preferences

    def update_preferences(self, prefs: Dict) -> None:
        """
        Merge new preferences into the stored preferences.

        Args:
            prefs: Partial preferences dict (only supplied keys are updated).
        """
        current = self.data["preferences"]
        for key, value in prefs.items():
            if key in current:
                # For list fields, extend rather than replace
                if isinstance(current[key], list) and isinstance(value, list):
                    merged = list(dict.fromkeys(current[key] + value))  # dedupe
                    current[key] = merged
                else:
                    current[key] = value

    # -------------------------------------------------- Context / Summary

    def get_context_summary(self) -> str:
        """
        Build a natural-language summary of the user's memory for LLM prompt injection.

        Returns:
            A multi-line string describing sessions, injuries, preferences, and stats.
        """
        parts: List[str] = []

        # --- Cumulative stats
        stats = self.data["cumulative_stats"]
        if stats["total_sessions"] > 0:
            parts.append(
                f"User has completed {stats['total_sessions']} coaching session(s) "
                f"with an average compliance rate of {stats['avg_compliance']:.0%}."
            )
            if stats["streak_current"] > 0:
                parts.append(
                    f"Current workout streak: {stats['streak_current']} day(s) "
                    f"(best: {stats['streak_best']})."
                )

        # --- Recent sessions
        recent = self.get_recent_sessions(3)
        if recent:
            summaries = [s["summary"] for s in recent if s.get("summary")]
            if summaries:
                parts.append(
                    "Recent session notes: " + " | ".join(summaries)
                )

        # --- Active injuries
        active = self.get_active_injuries()
        if active:
            injury_strs = [
                f"{i['body_part']} ({i['severity']})" for i in active
            ]
            parts.append(
                "Active injuries: " + ", ".join(injury_strs) + "."
            )

        # --- Preferences
        prefs = self.data["preferences"]
        pref_parts: List[str] = []
        if prefs.get("preferred_workout_types"):
            pref_parts.append(
                "preferred workouts: " + ", ".join(prefs["preferred_workout_types"])
            )
        if prefs.get("avoided_exercises"):
            pref_parts.append(
                "avoids: " + ", ".join(prefs["avoided_exercises"])
            )
        if prefs.get("goal"):
            pref_parts.append(f"goal: {prefs['goal']}")
        if pref_parts:
            parts.append("Preferences — " + "; ".join(pref_parts) + ".")

        return "\n".join(parts) if parts else "No prior history available."

    # -------------------------------------------------- Internal helpers

    def _update_cumulative_stats(self, session: Dict) -> None:
        """Recalculate cumulative stats after adding a session."""
        stats = self.data["cumulative_stats"]
        stats["total_sessions"] += 1

        compliance = session.get("compliance_rate", 0.0)
        # Running average
        n = stats["total_sessions"]
        stats["avg_compliance"] = (
            (stats["avg_compliance"] * (n - 1) + compliance) / n
        )

        # Streak logic: compliance >= 0.5 counts as a completed workout
        if compliance >= 0.5:
            stats["total_workouts_completed"] += 1
            stats["streak_current"] += 1
            if stats["streak_current"] > stats["streak_best"]:
                stats["streak_best"] = stats["streak_current"]
        else:
            stats["streak_current"] = 0
