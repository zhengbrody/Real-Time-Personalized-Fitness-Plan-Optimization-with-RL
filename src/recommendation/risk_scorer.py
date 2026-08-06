"""
Risk Scorer

Soft-constraint replacement for the legacy hard safety gate.
Aggregates weighted readiness / sleep / HRV / HR / fatigue / pain / soreness /
recent-training signals into a numeric score and a categorical risk_level, and
emits a structured list of RiskFactor objects that downstream modules
(session_planner, exercise_prescriber, orchestrator) use as the single source
of truth for building why_today, forbidden lists, and stop conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from src.validation.schemas import TodayRequest

Severity = Literal["small", "medium", "large"]
RiskLevel = Literal["low", "moderate", "elevated", "high"]


@dataclass
class RiskFactor:
    """Structured risk signal. `severity` drives forbidden/stop-condition logic.

    informational=True signals (e.g. "mild pain 2/10") are surfaced so downstream
    modules can still add them to forbidden / stop-condition lists, but they do
    NOT contribute to the numeric risk score -- a 2/10 twinge shouldn't push a
    borderline day over the 'elevated'/'high' boundary.
    """

    name: str
    severity: Severity
    note: str
    # Optional tags downstream modules can key off (e.g. affected body part).
    tags: List[str] = field(default_factory=list)
    informational: bool = False


# Severity -> score contribution. Tuned so a "normal but tired" day (one or two
# small/medium factors) lands in "moderate" rather than "elevated".
_SEVERITY_SCORE = {"small": 1.0, "medium": 2.0, "large": 4.0}


def _score_from_factors(factors: List[RiskFactor]) -> float:
    return sum(_SEVERITY_SCORE[f.severity] for f in factors if not f.informational)


def _risk_level_from_score(score: float) -> RiskLevel:
    if score < 2:
        return "low"
    if score < 5:
        return "moderate"
    if score < 8:
        return "elevated"
    return "high"


def _readiness_factor(readiness: Optional[float]) -> Optional[RiskFactor]:
    # readiness >= 65 is treated as "fine" -- we don't want a 68 reading
    # to push a tired-but-trainable day over the edge into "high".
    if readiness is None:
        return None
    if readiness < 30:
        return RiskFactor(
            name="readiness_very_low",
            severity="large",
            note=f"readiness {readiness:.0f} (<30)",
        )
    if readiness < 50:
        return RiskFactor(
            name="readiness_low",
            severity="medium",
            note=f"readiness {readiness:.0f} (<50)",
        )
    if readiness < 65:
        return RiskFactor(
            name="readiness_moderate",
            severity="small",
            note=f"readiness {readiness:.0f}",
        )
    return None


def _sleep_factor(hours: Optional[float]) -> Optional[RiskFactor]:
    if hours is None:
        return None
    if hours < 4.5:
        return RiskFactor(
            name="sleep_severe",
            severity="large",
            note=f"sleep {hours:.1f}h (<4.5)",
        )
    if hours < 6:
        return RiskFactor(
            name="sleep_low",
            severity="medium",
            note=f"sleep {hours:.1f}h (<6)",
        )
    if hours < 7:
        return RiskFactor(
            name="sleep_mild",
            severity="small",
            note=f"sleep {hours:.1f}h",
        )
    return None


def _hrv_factor(hrv: Optional[float]) -> Optional[RiskFactor]:
    # No personal baseline available -> absolute thresholds.
    if hrv is None:
        return None
    if hrv < 20:
        return RiskFactor(
            name="hrv_very_low",
            severity="large",
            note=f"HRV {hrv:.0f}ms (<20)",
        )
    if hrv < 30:
        return RiskFactor(
            name="hrv_low",
            severity="medium",
            note=f"HRV {hrv:.0f}ms (<30)",
        )
    if hrv < 40:
        return RiskFactor(
            name="hrv_mild",
            severity="small",
            note=f"HRV {hrv:.0f}ms",
        )
    return None


def _resting_hr_factor(rhr: Optional[int]) -> Optional[RiskFactor]:
    if rhr is None:
        return None
    if rhr > 90:
        return RiskFactor(
            name="resting_hr_very_high",
            severity="large",
            note=f"resting HR {rhr}bpm (>90)",
        )
    if rhr > 75:
        return RiskFactor(
            name="resting_hr_elevated",
            severity="small",
            note=f"resting HR {rhr}bpm (>75)",
        )
    return None


def _fatigue_factor(fatigue: Optional[int]) -> Optional[RiskFactor]:
    if fatigue is None:
        return None
    if fatigue >= 9:
        return RiskFactor(
            name="fatigue_severe",
            severity="large",
            note=f"fatigue {fatigue}/10",
        )
    if fatigue >= 7:
        return RiskFactor(
            name="fatigue_high",
            severity="medium",
            note=f"fatigue {fatigue}/10",
        )
    return None


def _pain_factors(pain: dict) -> List[RiskFactor]:
    out: List[RiskFactor] = []
    for part, level in (pain or {}).items():
        if level is None:
            continue
        if level >= 7:
            out.append(
                RiskFactor(
                    name=f"pain_severe:{part}",
                    severity="large",
                    note=f"{part} pain {level}/10",
                    tags=[part],
                )
            )
        elif level >= 4:
            out.append(
                RiskFactor(
                    name=f"pain:{part}",
                    severity="medium",
                    note=f"{part} pain {level}/10",
                    tags=[part],
                )
            )
        elif level >= 2:
            out.append(
                RiskFactor(
                    name=f"pain_mild:{part}",
                    severity="small",
                    note=f"{part} pain {level}/10",
                    tags=[part],
                    informational=True,
                )
            )
    return out


def _soreness_factors(soreness: dict) -> List[RiskFactor]:
    out: List[RiskFactor] = []
    for group, level in (soreness or {}).items():
        if level is None:
            continue
        if level >= 7:
            out.append(
                RiskFactor(
                    name=f"soreness_high:{group}",
                    severity="medium",
                    note=f"{group} soreness {level}/10",
                    tags=[group],
                )
            )
        elif level >= 5:
            out.append(
                RiskFactor(
                    name=f"soreness:{group}",
                    severity="small",
                    note=f"{group} soreness {level}/10",
                    tags=[group],
                )
            )
    return out


def _recent_training_factors(recent: list) -> List[RiskFactor]:
    """Residual fatigue from recent sessions.

    - A high-RPE (>=8.5) session within the last 48h -> +score.
    - 2+ consecutive hard days (rpe>=7.5 on days_ago 1 and 2) -> +score.
    """
    out: List[RiskFactor] = []
    if not recent:
        return out

    for s in recent:
        if s.days_ago <= 2 and s.rpe is not None and s.rpe >= 8.5:
            out.append(
                RiskFactor(
                    name="recent_high_rpe",
                    severity="medium",
                    note=f"{s.type} RPE {s.rpe} {s.days_ago}d ago",
                    tags=[s.type.lower()],
                )
            )

    # consecutive hard days
    by_day = {s.days_ago: s for s in recent}
    d1, d2 = by_day.get(1), by_day.get(2)
    if (
        d1 is not None
        and d2 is not None
        and d1.rpe is not None
        and d2.rpe is not None
        and d1.rpe >= 7.5
        and d2.rpe >= 7.5
    ):
        out.append(
            RiskFactor(
                name="consecutive_hard_days",
                severity="small",
                note=f"back-to-back hard days ({d2.type} then {d1.type})",
            )
        )
    return out


def _collect_data_gaps(req: TodayRequest) -> List[str]:
    gaps: List[str] = []
    w = req.wearable
    if w.readiness_score is None:
        gaps.append("no readiness_score")
    if w.sleep_hours is None:
        gaps.append("no sleep_hours")
    if w.hrv is None:
        gaps.append("no HRV")
    if w.resting_hr is None:
        gaps.append("no resting_hr")
    m = req.manual
    if m.fatigue is None:
        gaps.append("no subjective fatigue")
    if not m.pain:
        gaps.append("no pain check-in")
    if not m.soreness:
        gaps.append("no soreness check-in")
    if not req.recent_training:
        gaps.append("no recent_training history")
    return gaps


def score(
    request: TodayRequest,
) -> Tuple[List[RiskFactor], RiskLevel, List[str]]:
    """Compute risk factors, risk level, and data gaps.

    Missing signals are skipped (graceful degradation) and recorded in data_gaps
    so the downstream layer can communicate reduced confidence.
    """
    factors: List[RiskFactor] = []

    for f in (
        _readiness_factor(request.wearable.readiness_score),
        _sleep_factor(request.wearable.sleep_hours),
        _hrv_factor(request.wearable.hrv),
        _resting_hr_factor(request.wearable.resting_hr),
        _fatigue_factor(request.manual.fatigue),
    ):
        if f is not None:
            factors.append(f)

    factors.extend(_pain_factors(request.manual.pain))
    factors.extend(_soreness_factors(request.manual.soreness))
    factors.extend(_recent_training_factors(request.recent_training))

    total = _score_from_factors(factors)
    level = _risk_level_from_score(total)
    gaps = _collect_data_gaps(request)
    return factors, level, gaps


def is_extreme_stack(factors: List[RiskFactor]) -> bool:
    """True iff the risk profile is severe enough to justify collapsing to rest.

    Per design: only hard-block when the state is genuinely dangerous. Stage-1
    definition: 3+ large-severity factors.
    """
    large = [f for f in factors if f.severity == "large"]
    return len(large) >= 3
