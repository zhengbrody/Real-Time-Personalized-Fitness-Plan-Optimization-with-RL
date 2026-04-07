"""
AI Coach Agent Module

This module implements a tool-using AI agent that:
- Explains training plan recommendations
- Collects user feedback (RPE, mood, pain, stress)
- Triggers actions (plan adjustments, recovery scheduling, summaries)
- Enables closed-loop learning through feedback events
"""

from .coach_agent import CoachAgent
from .tools import AgentTools
from .safety import SafetyGuardrails
from .state import DailyState
from .memory import CoachMemory
from .session_handoff import session_handoff

__all__ = [
    "CoachAgent",
    "AgentTools",
    "SafetyGuardrails",
    "DailyState",
    "CoachMemory",
    "session_handoff",
]
