"""
Session Handoff — Summarises a coaching session and persists it to memory.

This module provides a pure-Python (no LLM) function that distills a session's
conversation, recommendations, and feedback into a compact summary, writes it
to CoachMemory, and saves to disk.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .memory import CoachMemory


def session_handoff(
    memory: CoachMemory,
    *,
    recommendations: Optional[List[str]] = None,
    feedback: Optional[Dict] = None,
    mood: Optional[str] = None,
    compliance_rate: float = 0.0,
    plan: Optional[Dict] = None,
    tools_called: Optional[List[Dict]] = None,
) -> Dict:
    """
    Summarise the current session and persist it to memory.

    Args:
        memory: CoachMemory instance (must already be loaded).
        recommendations: List of recommendation strings given this session.
        feedback: User feedback dict (RPE, mood, etc.).
        mood: Short mood label (e.g. "good", "tired", "stressed").
        compliance_rate: 0.0–1.0 how well the user followed the plan.
        plan: The training plan dict served this session.
        tools_called: List of tool-call result dicts from the agent.

    Returns:
        The session summary dict that was stored.
    """
    recommendations = recommendations or []
    feedback = feedback or {}
    tools_called = tools_called or []

    summary = _build_summary(
        recommendations=recommendations,
        feedback=feedback,
        mood=mood,
        plan=plan,
        tools_called=tools_called,
    )

    session_data = {
        "session_id": str(uuid.uuid4()),
        "date": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "recommendations_given": recommendations,
        "feedback_received": feedback,
        "mood": mood or "",
        "compliance_rate": compliance_rate,
    }

    memory.add_session(session_data)
    memory.save()

    return session_data


# ---------------------------------------------------------------- helpers


def _build_summary(
    *,
    recommendations: List[str],
    feedback: Dict,
    mood: Optional[str],
    plan: Optional[Dict],
    tools_called: List[Dict],
) -> str:
    """
    Rule-based session summariser (no LLM needed).

    Combines plan details, recommendations, feedback, and mood into a single
    concise string.
    """
    parts: List[str] = []

    # Plan info
    if plan:
        intensity = plan.get("intensity", "unknown")
        parts.append(f"Plan intensity: {intensity}.")

    # Recommendations
    if recommendations:
        parts.append(f"Gave {len(recommendations)} recommendation(s).")

    # Mood
    if mood:
        parts.append(f"Mood: {mood}.")

    # Feedback highlights
    if feedback:
        rpe = feedback.get("rpe")
        if rpe is not None:
            parts.append(f"RPE: {rpe}.")
        pain = feedback.get("pain")
        if pain:
            parts.append(f"Pain reported: {pain}.")

    # Tools used
    if tools_called:
        tool_names = [t.get("tool", "unknown") for t in tools_called]
        parts.append(f"Tools used: {', '.join(tool_names)}.")

    return " ".join(parts) if parts else "Session completed with no notable events."
