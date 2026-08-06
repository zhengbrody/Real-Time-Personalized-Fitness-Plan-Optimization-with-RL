"""
Hybrid Recommender (Stage-1 refactor)

New entry point is `recommend(request: TodayRequest) -> Recommendation`, which
orchestrates:

    risk_scorer -> session_planner -> exercise_prescriber

and wraps the results plus derived fields (volume_cap, forbidden list,
override option, stop conditions, data gaps) into the dual-layer
Recommendation output.

The legacy dict-based `recommend(state)` path is kept as a thin shim on the
`HybridRecommender` class so existing callers / tests still return something
sensible (a dict built from the new Recommendation) until Stage 2 updates
them. The old action-space / contextual bandit / rule-based path is left
intact and available via `recommend_legacy()` for tests that inspect the old
single-action shape.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.validation.schemas import (
    ExerciseRx,
    Recommendation,
    SessionPlan,
    TodayDecision,
    TodayRequest,
    WearableData,
    ManualCheckIn,
)

from . import risk_scorer, session_planner, exercise_prescriber
from .action_space import Action, ActionSpace
from .contextual_bandits import ContextualBandit
from .reward_fn import RewardFunction
from src.safety.safety_gate import SafetyGate

# ---------------------------------------------------------------------------
# Stage-1 public entry point.
# ---------------------------------------------------------------------------


def _derive_volume_cap(decision: TodayDecision, factors) -> str:
    intensity = decision.recommended_intensity
    if intensity == "hard":
        return "total hard sets <= 22"
    if intensity == "moderate":
        return "total hard sets <= 18; no main lift above 80% 1RM"
    if intensity == "light":
        return "total hard sets <= 14; no main lift above 70% 1RM"
    if intensity == "recovery":
        return "total hard sets <= 8; keep RPE <= 6"
    return "no loaded sets today"


def _derive_forbidden(request: TodayRequest, factors) -> List[str]:
    out: List[str] = []
    pain = {
        k.lower(): v for k, v in (request.manual.pain or {}).items() if v is not None
    }
    for part, level in pain.items():
        if level >= 4:
            if "knee" in part:
                out.append(
                    f"heavy bilateral squats / max-effort lunges ({part} pain {level}/10)"
                )
            elif "back" in part:
                out.append(
                    f"conventional deadlift / heavy bent row ({part} pain {level}/10)"
                )
            elif "shoulder" in part:
                out.append(f"overhead press, heavy bench ({part} pain {level}/10)")
            elif "elbow" in part or "wrist" in part:
                out.append(
                    f"max loaded pressing / supinated curls ({part} pain {level}/10)"
                )
            else:
                out.append(f"loaded movements aggravating {part} (pain {level}/10)")
        elif level >= 2:
            # Below the "avoid" threshold we still flag the movement as "not recommended heavy".
            if "knee" in part:
                out.append(
                    f"heavy bilateral barbell squat / deep lunges ({part} pain {level}/10 - avoid heavy, light ok)"
                )
            elif "back" in part:
                out.append(
                    f"max-effort deadlift ({part} pain {level}/10 - avoid heavy, light ok)"
                )

    # recent-heavy overlap: protect already-trashed groups from a second beating.
    for f in factors:
        if f.name == "recent_high_rpe":
            out.append(
                "repeating yesterday's heavy focus at high load (let it recover)"
            )
            break

    return out


def _derive_override(request: TodayRequest, decision: TodayDecision, factors) -> str:
    """What the user may push to if they insist; includes a ceiling."""
    pref = request.override_preference or "follow_recommendation"
    intensity = decision.recommended_intensity

    ceilings = {
        "hard": "RPE 9",
        "moderate": "RPE 8",
        "light": "RPE 7-8 on accessories only",
        "recovery": "RPE 6 on non-painful movements",
        "rest": "10-20min easy walk; no loading",
    }
    base_ceiling = ceilings[intensity]

    if pref == "go_heavier_if_possible":
        # Allow a one-notch bump but keep the ceiling explicit.
        bump = {
            "recovery": "move to light session, ceiling RPE 7 on pain-free movements",
            "light": "move to moderate session, ceiling RPE 7-8 on non-painful lifts",
            "moderate": "push top-set to RPE 8, keep accessories as prescribed",
            "hard": "push top-set to RPE 9 if all warm-ups feel crisp",
            "rest": "optional 20min zone-2 + mobility only",
        }
        return bump[intensity]
    if pref == "go_lighter":
        lighter = {
            "hard": "drop to moderate session, ceiling RPE 7",
            "moderate": "drop to light session, ceiling RPE 6-7",
            "light": "drop to recovery session, mobility + zone-2 only",
            "recovery": "convert to full rest day",
            "rest": "rest day (as planned)",
        }
        return lighter[intensity]
    return f"may push to {base_ceiling} if warm-ups feel crisp; otherwise stick to plan"


def _derive_stop_conditions(factors) -> List[str]:
    """Dynamic stop conditions built from the same RiskFactor source of truth."""
    conds: List[str] = [
        "sharp joint pain (any location)",
        "dizziness, chest tightness, or abnormal HR response",
        "form breakdown on main lifts",
    ]
    seen_parts = set()
    for f in factors:
        if f.name.startswith("pain") and f.tags:
            part = f.tags[0]
            if part in seen_parts:
                continue
            seen_parts.add(part)
            conds.append(f"any increase in {part} pain during or after warm-up sets")
        if f.name == "hrv_very_low" or f.name == "resting_hr_very_high":
            conds.append("HR > 180 bpm sustained >30s during conditioning")
        if f.name == "fatigue_severe":
            conds.append("sudden heavy fatigue / lightheadedness - cut session")
    return conds


def recommend(request: TodayRequest) -> Recommendation:
    """Stage-1 entry point. Produces the full dual-layer Recommendation."""
    factors, risk_level, data_gaps = risk_scorer.score(request)
    today_decision, template = session_planner.plan(request, risk_level, factors)
    exercises = exercise_prescriber.prescribe(template, today_decision, request)

    session_plan = SessionPlan(
        warmup=list(template.warmup),
        main_lifts=list(template.main_lifts),
        accessories=list(template.accessories),
        conditioning=list(template.conditioning),
        cooldown=list(template.cooldown),
    )

    return Recommendation(
        today_decision=today_decision,
        session_plan=session_plan,
        exercise_prescription=exercises,
        volume_cap=_derive_volume_cap(today_decision, factors),
        forbidden_or_not_recommended=_derive_forbidden(request, factors),
        override_option=_derive_override(request, today_decision, factors),
        stop_conditions=_derive_stop_conditions(factors),
        data_gaps=data_gaps,
    )


# ---------------------------------------------------------------------------
# Legacy class-based facade. Kept for backward compatibility with existing
# callers / tests that still use dict-state + ActionSpace + ContextualBandit.
# ---------------------------------------------------------------------------


def _legacy_state_to_request(state: Dict) -> TodayRequest:
    wearable = WearableData(
        readiness_score=state.get("readiness_score"),
        sleep_score=state.get("sleep_score"),
        sleep_hours=state.get("sleep_duration_hours"),
        hrv=state.get("hrv"),
        resting_hr=state.get("resting_hr"),
        activity_score=state.get("activity_score"),
    )
    manual_pain = state.get("pain") if isinstance(state.get("pain"), dict) else {}
    manual_soreness = (
        state.get("soreness") if isinstance(state.get("soreness"), dict) else {}
    )
    manual = ManualCheckIn(
        fatigue=state.get("fatigue"),
        motivation=state.get("motivation"),
        pain=manual_pain or {},
        soreness=manual_soreness or {},
    )
    return TodayRequest(wearable=wearable, manual=manual)


class HybridRecommender:
    """Hybrid recommendation facade.

    Stage-1 behavior:
      - `recommend(state)` on a dict returns a dict shim built from the new
        Recommendation (key fields copied to match legacy-ish surface).
      - `recommend(TodayRequest)` returns the full Recommendation pydantic model.
      - The old single-action / bandit path is still available via
        `recommend_legacy()`.
    """

    def __init__(self, use_rl: bool = True):
        self.action_space = ActionSpace()
        self.reward_fn = RewardFunction()
        self.safety_gate = SafetyGate()

        if use_rl:
            self.bandit = ContextualBandit(self.action_space, self.reward_fn)
        else:
            self.bandit = None

    # ------------------------------------------------------------------
    # New dual-layer entry points.
    # ------------------------------------------------------------------

    def recommend(self, state, use_rl: Optional[bool] = None):
        """Dispatch based on input type.

        - TodayRequest -> new dual-layer Recommendation.
        - dict        -> legacy-style dict (built from the new Recommendation)
                          so existing consumers keep working with minimal churn.
        """
        if isinstance(state, TodayRequest):
            return recommend(state)

        # dict path: produce a shim dict with the new data embedded.
        request = _legacy_state_to_request(state)
        rec = recommend(request)

        # Approximate mapping back to legacy action-space-style fields to keep
        # old callers alive. Exact action_id is best-effort.
        legacy_intensity_map = {
            "rest": "NONE",
            "recovery": "LOW",
            "light": "LOW",
            "moderate": "MEDIUM",
            "hard": "HIGH",
        }
        legacy_type_map = {
            "rest": "REST",
            "recovery": "RECOVERY",
            "light": "STRENGTH",
            "moderate": "STRENGTH",
            "hard": "STRENGTH",
        }
        intensity_legacy = legacy_intensity_map[
            rec.today_decision.recommended_intensity
        ]
        wtype_legacy = legacy_type_map[rec.today_decision.recommended_intensity]
        duration = state.get("time_budget_min") or request.time_budget_min

        # Best-effort back-map to a legacy action_id so old API/test clients that
        # expect an int in [0, 17] keep working.
        legacy_duration = 30
        legacy_action_id = self.action_space.get_action_id(
            wtype_legacy, intensity_legacy, legacy_duration
        )

        return {
            "action_id": legacy_action_id,
            "workout_type": wtype_legacy,
            "intensity": intensity_legacy,
            "duration_minutes": duration,
            "description": rec.today_decision.primary_goal_today,
            "safety_check": {
                "is_safe": rec.today_decision.risk_level in ("low", "moderate"),
                "message": rec.today_decision.why_today,
            },
            "rationale": rec.today_decision.why_today,
            # Expose the full new recommendation so downstream can start
            # consuming it incrementally.
            "recommendation": rec.model_dump(),
        }

    # ------------------------------------------------------------------
    # Legacy single-action path. Preserved so bandit-specific tests still pass.
    # ------------------------------------------------------------------

    def recommend_legacy(self, state: Dict, use_rl: Optional[bool] = None) -> Dict:
        use_rl = use_rl if use_rl is not None else (self.bandit is not None)

        safety_result = self.safety_gate.check_state(state)

        all_action_ids = list(range(self.action_space.get_action_count()))
        allowed_actions = self.safety_gate.filter_actions(state, all_action_ids)
        if not allowed_actions:
            allowed_actions = [0]

        if use_rl and self.bandit:
            context = self._state_to_context(state)
            action_id = self.bandit.select_action(context, allowed_actions)
        else:
            action_id = self._rule_based_recommendation(state, allowed_actions)

        action = self.action_space.get_action(action_id)
        return {
            "action_id": action_id,
            "workout_type": action.workout_type,
            "intensity": action.intensity,
            "duration_minutes": action.duration_minutes,
            "description": action.description,
            "safety_check": {
                "is_safe": safety_result.is_safe,
                "message": safety_result.message,
            },
            "rationale": self._generate_rationale(state, action),
        }

    def _state_to_context(self, state: Dict) -> np.ndarray:
        features = [
            state.get("readiness_score", 50) / 100.0,
            state.get("sleep_score", 50) / 100.0,
            state.get("activity_score", 50) / 100.0,
            state.get("hrv", 50) / 100.0,
            state.get("resting_hr", 60) / 100.0,
            state.get("fatigue", 5) / 10.0,
            state.get("days_since_training", 1) / 7.0,
        ]
        return np.array(features)

    def _rule_based_recommendation(
        self, state: Dict, allowed_actions: List[int]
    ) -> int:
        readiness = state.get("readiness_score", 50)
        sleep_hours = state.get("sleep_duration_hours", 7)
        fatigue = state.get("fatigue", 5)
        days_since = state.get("days_since_training", 1)

        if readiness < 40 or sleep_hours < 5:
            return 0

        if fatigue >= 7:
            recovery_actions = [
                a
                for a in allowed_actions
                if self.action_space.get_action(a).workout_type == "RECOVERY"
            ]
            if recovery_actions:
                return recovery_actions[0]

        if days_since >= 3:
            medium_actions = [
                a
                for a in allowed_actions
                if self.action_space.get_action(a).intensity == "MEDIUM"
            ]
            if medium_actions:
                return medium_actions[0]

        low_actions = [
            a
            for a in allowed_actions
            if self.action_space.get_action(a).intensity == "LOW"
        ]
        if low_actions:
            return low_actions[0]

        return allowed_actions[0]

    def _generate_rationale(self, state: Dict, action: Action) -> str:
        readiness = state.get("readiness_score", 50)
        sleep_hours = state.get("sleep_duration_hours", 7)

        if action.workout_type == "REST":
            return (
                f"Rest day recommended due to low readiness ({readiness}) or "
                f"insufficient sleep ({sleep_hours:.1f}h)"
            )
        elif action.intensity == "LOW":
            return (
                f"Low intensity {action.workout_type.lower()} recommended based on "
                "current recovery state"
            )
        else:
            return (
                f"{action.intensity} intensity {action.workout_type.lower()} "
                "recommended - you're well recovered"
            )

    def update(self, action_id: int, state: Dict, reward: float):
        if self.bandit and action_id is not None:
            self.bandit.update(action_id, reward)
