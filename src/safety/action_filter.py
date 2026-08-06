"""
Canonical Safety Action Filter
==============================

The single implementation of "which actions is it safe to offer today".

Why this module exists
----------------------
The rules used to live in two places: :class:`~src.safety.safety_gate.SafetyGate`
for the serving path, and a private ``_MinimalSafetyFilter`` copied into the
benchmark script so it could avoid a heavy import chain.  The two drifted — the
benchmark never applied the consecutive-hard-days rule — which meant the policy
was benchmarked against a *different* constraint set than the one it runs under
in production.  Every number measured that way is measured on the wrong problem.

So the rules live here, once.  The gate delegates to it, the benchmark imports
it, and ``tests/test_feature_parity.py`` asserts the two paths agree.

Properties this module guarantees
---------------------------------
- **Pure.**  No I/O, no clock, no globals.  Same state in, same list out.
- **Fail-closed.**  A missing field falls back to the *conservative* value, not
  the optimistic one: an absent fatigue reading is treated as elevated, not as
  zero.  A recommender that silently gets safer when data is missing is
  correct; one that silently gets bolder is dangerous.
- **Never empty.**  REST (action 0) is always permitted, so the bandit always
  has at least one legal choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from src.recommendation.action_space import ActionSpace

__all__ = ["SafetyDecision", "filter_actions", "explain"]

_INTENSITY_ORDER: Dict[str, int] = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

# Conservative fallbacks, applied when a field is missing or unparseable.
_FALLBACK_READINESS = 45.0  # below the "fresh" band
_FALLBACK_FATIGUE = 6.5  # above the "reduce intensity" threshold
_FALLBACK_SLEEP_H = 6.0
_FALLBACK_HRV = 35.0
_FALLBACK_ACWR = 1.0


@dataclass
class SafetyDecision:
    """The filtered action set plus the reasons it was narrowed."""

    allowed_action_ids: List[int]
    max_intensity: str
    allowed_types: List[str]
    triggered_rules: List[str] = field(default_factory=list)

    @property
    def is_restricted(self) -> bool:
        return bool(self.triggered_rules)


def _num(state: Mapping[str, Any], key: str, fallback: float) -> float:
    value = state.get(key, None)
    if value is None:
        return fallback
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if out != out:  # NaN
        return fallback
    return out


def filter_actions(
    state: Mapping[str, Any],
    action_space: Optional[ActionSpace] = None,
) -> SafetyDecision:
    """
    Narrow the action space to what is physiologically safe for ``state``.

    Parameters
    ----------
    state
        Raw body-state record (the same shape the feature transform consumes).
    action_space
        Defaults to the standard 18-action space.

    Returns
    -------
    SafetyDecision
        Allowed action ids and the rules that fired.
    """
    space = action_space or ActionSpace()

    readiness = _num(state, "readiness_score", _FALLBACK_READINESS)
    fatigue = _num(state, "fatigue", _FALLBACK_FATIGUE)
    sleep_h = _num(state, "sleep_duration_hours", _FALLBACK_SLEEP_H)
    hrv = _num(state, "hrv", _FALLBACK_HRV)
    consec_hard = _num(state, "consecutive_hard_days", 0.0)
    acwr = _num(state, "acwr", _FALLBACK_ACWR)
    soreness = _num(state, "soreness", 0.0)

    allowed_types = ["REST", "RECOVERY", "STRENGTH", "CARDIO"]
    max_intensity = "HIGH"
    triggered: List[str] = []

    def _restrict(intensity: str) -> None:
        nonlocal max_intensity
        if _INTENSITY_ORDER[intensity] < _INTENSITY_ORDER[max_intensity]:
            max_intensity = intensity

    # -- critical rules: training is off the table entirely ---------------
    if readiness < 30.0 or fatigue > 8.0:
        allowed_types = ["REST", "RECOVERY"]
        _restrict("LOW")
        triggered.append("critical_depletion")

    if sleep_h < 4.0 or hrv < 20.0:
        allowed_types = ["REST", "RECOVERY"]
        _restrict("LOW")
        triggered.append("low_hrv_or_sleep")

    if soreness >= 8.0:
        allowed_types = ["REST", "RECOVERY"]
        _restrict("LOW")
        triggered.append("high_soreness")

    # -- graded rules: training allowed, intensity capped -----------------
    if fatigue > 6.0:
        _restrict("LOW")
        triggered.append("elevated_fatigue")

    if consec_hard >= 3.0:
        _restrict("MEDIUM")
        triggered.append("consecutive_high_load")

    if acwr > 1.5:
        _restrict("MEDIUM")
        triggered.append("acute_chronic_spike")

    allowed = space.filter_by_safety(
        allowed_types=allowed_types, max_intensity=max_intensity
    )

    # Fail-safe: the policy must always have a legal action.
    if not allowed:
        allowed = [0]

    return SafetyDecision(
        allowed_action_ids=sorted(allowed),
        max_intensity=max_intensity,
        allowed_types=allowed_types,
        triggered_rules=triggered,
    )


def explain(decision: SafetyDecision) -> str:
    """Human-readable summary, used in API responses and coach explanations."""
    if not decision.is_restricted:
        return "No safety restrictions: full action space available."
    reasons = {
        "critical_depletion": "readiness or fatigue at a critical level",
        "low_hrv_or_sleep": "HRV or sleep below the safe threshold",
        "high_soreness": "muscle soreness too high to load",
        "elevated_fatigue": "elevated fatigue caps intensity at LOW",
        "consecutive_high_load": "three or more consecutive hard days",
        "acute_chronic_spike": "acute:chronic workload ratio above 1.5",
    }
    parts = [reasons.get(r, r) for r in decision.triggered_rules]
    return (
        f"Restricted to {decision.max_intensity} intensity "
        f"({', '.join(decision.allowed_types)}): " + "; ".join(parts) + "."
    )
