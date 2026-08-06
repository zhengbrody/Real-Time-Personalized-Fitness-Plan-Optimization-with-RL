"""
Safety Gate

Hard rules to prevent dangerous recommendations.

NOTE (Stage 1 refactor): the hard-gate behavior (`check_state`, `filter_actions`)
is retained for backward compatibility with legacy callers, but is deprecated.
New code should go through `src/recommendation/risk_scorer.py`, which produces
a soft risk score + structured RiskFactor list. `compute_risk_factors()` below
is the thin adapter the new pipeline uses when starting from the legacy
dict-based state shape.
"""

from typing import Dict, List
from dataclasses import dataclass

from src.agent.safety import SafetyGuardrails, SafetyCheckResult


@dataclass
class SafetyRule:
    """Safety rule definition."""

    name: str
    condition: str
    max_intensity: str
    allowed_types: List[str]
    message: str


class SafetyGate:
    """
    Safety gate for action filtering.

    Prevents dangerous recommendations before they reach the bandit.
    """

    def __init__(self):
        """Initialize safety gate."""
        self.rules = self._create_safety_rules()
        self.safety_guardrails = SafetyGuardrails()

    def _create_safety_rules(self) -> List[SafetyRule]:
        """Create safety rules."""
        return [
            SafetyRule(
                name="low_hrv_low_sleep",
                condition="hrv < 20 or sleep < 4 hours",
                max_intensity="LOW",
                allowed_types=["REST", "RECOVERY"],
                message="Low HRV and/or insufficient sleep - rest day recommended",
            ),
            SafetyRule(
                name="high_fatigue",
                condition="fatigue >= 8",
                max_intensity="LOW",
                allowed_types=["REST", "RECOVERY"],
                message="High fatigue level - reduce intensity",
            ),
            SafetyRule(
                name="high_soreness",
                condition="soreness >= 8",
                max_intensity="LOW",
                allowed_types=["REST", "RECOVERY"],
                message="High muscle soreness - active recovery only",
            ),
            SafetyRule(
                name="consecutive_high_load",
                condition="consecutive_high_days >= 3",
                max_intensity="MEDIUM",
                allowed_types=["REST", "RECOVERY", "STRENGTH", "CARDIO"],
                message="Consecutive high-intensity days - reduce load",
            ),
        ]

    def check_state(self, state: Dict) -> SafetyCheckResult:
        """
        Check state against safety rules.

        DEPRECATED (Stage 1): prefer `risk_scorer.score()` in new code paths.
        Kept alive so legacy callers (hybrid_recommender shim, existing tests)
        continue to work until Stage 2 removes hard-gate usage.

        Args:
            state: Daily state dictionary

        Returns:
            SafetyCheckResult
        """
        # Use existing safety guardrails
        from src.agent.state import DailyState

        daily_state = DailyState(
            user_id=state.get("user_id", "unknown"),
            date=state.get("date", ""),
            hrv=state.get("hrv"),
            resting_heart_rate=state.get("resting_hr"),
            sleep_duration_hours=state.get("sleep_duration_hours"),
            sleep_quality_score=state.get("sleep_score"),
            readiness_score=state.get("readiness_score"),
            fatigue_level=state.get("fatigue"),
            muscle_soreness=state.get("soreness"),
            overtraining_risk=state.get("overtraining_risk", False),
        )

        return self.safety_guardrails.check_state(daily_state)

    def filter_actions(self, state: Dict, all_action_ids: List[int]) -> List[int]:
        """
        Filter actions based on safety rules.

        Args:
            state: Daily state
            all_action_ids: All possible action IDs

        Returns:
            List of safe action IDs
        """
        safety_result = self.check_state(state)

        if not safety_result.is_safe:
            # Get max intensity and allowed types from safety result
            if safety_result.recommended_action == "mandatory_rest_day":
                # Only allow REST
                from src.recommendation.action_space import ActionSpace

                action_space = ActionSpace()
                return [0]  # REST action ID

            elif safety_result.recommended_action == "rest_day_or_light_activity":
                # Allow REST and RECOVERY only
                from src.recommendation.action_space import ActionSpace

                action_space = ActionSpace()
                return action_space.filter_by_safety(
                    allowed_types=["REST", "RECOVERY"], max_intensity="LOW"
                )

            elif safety_result.recommended_action == "reduce_intensity":
                # Reduce max intensity
                from src.recommendation.action_space import ActionSpace

                action_space = ActionSpace()
                return action_space.filter_by_safety(
                    allowed_types=["REST", "RECOVERY", "STRENGTH", "CARDIO"],
                    max_intensity="MEDIUM",
                )

        # All actions allowed
        return all_action_ids

    def compute_risk_factors(self, state):
        """
        Adapter: accept either a dict-style state or a TodayRequest and return
        the new soft RiskFactor list plus the derived risk_level and data_gaps.

        This is the hook documented in the Stage-1 refactor: downstream modules
        that still only have a dict/BodyState in hand can call through this
        method, but internally it delegates to
        `src.recommendation.risk_scorer.score`, which is the single source of
        truth for the new soft-constraint scoring system.
        """
        # Local imports to avoid a circular dependency at module-load time.
        from src.recommendation.risk_scorer import score as _risk_score
        from src.validation.schemas import (
            ManualCheckIn,
            TodayRequest,
            WearableData,
        )

        if isinstance(state, TodayRequest):
            req = state
        else:
            wearable = WearableData(
                readiness_score=state.get("readiness_score"),
                sleep_score=state.get("sleep_score"),
                sleep_hours=state.get("sleep_duration_hours"),
                hrv=state.get("hrv"),
                resting_hr=state.get("resting_hr"),
                activity_score=state.get("activity_score"),
            )
            manual = ManualCheckIn(
                fatigue=state.get("fatigue"),
                pain=state.get("pain", {}) or {},
                soreness=(
                    state.get("soreness")
                    if isinstance(state.get("soreness"), dict)
                    else {}
                ),
            )
            req = TodayRequest(wearable=wearable, manual=manual)

        return _risk_score(req)
