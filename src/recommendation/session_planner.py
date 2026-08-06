"""
Session Planner

Given risk_level + goal + equipment + recent training + risk factors, pick a
primary_goal_today, a recommended_intensity, and a session template
(warmup/main/accessories/conditioning/cooldown).

Design priority: the user defaults to training. Moderate fatigue should
downgrade volume/intensity, not collapse to "rest". Only stack of extreme
signals (or explicit overtraining flag) produces a true rest day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from src.validation.schemas import TodayDecision, TodayRequest
from .risk_scorer import RiskFactor, RiskLevel, is_extreme_stack

Intensity = Literal["rest", "recovery", "light", "moderate", "hard"]


@dataclass
class SessionTemplate:
    """Rule-based template. Names here get expanded into ExerciseRx downstream."""

    key: str
    primary_goal: str
    warmup: List[str]
    main_lifts: List[str]
    accessories: List[str]
    conditioning: List[str]
    cooldown: List[str]
    # Muscle groups / joints this template stresses heavily; used by the selector
    # to avoid overlap with recent work and to honor pain constraints.
    stresses: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Template library. Intentionally small (6-8 templates) -- covers push/pull/
# lower/full-body plus two non-strength modalities for elevated/high risk days.
# ----------------------------------------------------------------------------

_TEMPLATES: Dict[str, SessionTemplate] = {
    "upper_push": SessionTemplate(
        key="upper_push",
        primary_goal="upper body push strength",
        warmup=["5min row or bike easy", "band pull-aparts 2x15", "scap push-ups 2x10"],
        main_lifts=["bench press", "overhead press"],
        accessories=["incline db press", "dips or push-ups", "triceps extension"],
        conditioning=[],
        cooldown=["chest + lat stretch 3min", "thoracic extensions 2x8"],
        stresses=["chest", "shoulders", "triceps", "upper_body"],
    ),
    "upper_pull": SessionTemplate(
        key="upper_pull",
        primary_goal="upper body pull strength",
        warmup=["5min row easy", "band pull-aparts 2x15", "dead hangs 2x20s"],
        main_lifts=["weighted pull-up", "barbell row"],
        accessories=["lat pulldown", "face pull", "biceps curl"],
        conditioning=[],
        cooldown=["lat + forearm stretch 3min"],
        stresses=["back", "biceps", "rear_delts", "upper_body"],
    ),
    "upper_full": SessionTemplate(
        key="upper_full",
        primary_goal="upper body balanced (push+pull), leg-sparing",
        warmup=["5min row easy", "band pull-aparts 2x15", "scap push-ups 2x10"],
        main_lifts=["overhead press", "chest-supported row"],
        accessories=[
            "incline db press",
            "lat pulldown",
            "face pull",
            "triceps extension",
            "hammer curl",
        ],
        conditioning=[],
        cooldown=["thoracic mobility 3min", "neck + shoulder stretch"],
        stresses=["chest", "back", "shoulders", "arms", "upper_body"],
    ),
    "lower": SessionTemplate(
        key="lower",
        primary_goal="lower body strength",
        warmup=[
            "5min bike easy",
            "bodyweight squats 2x10",
            "hip openers 2x8/side",
        ],
        main_lifts=["back squat", "romanian deadlift"],
        accessories=["bulgarian split squat", "leg curl", "standing calf raise"],
        conditioning=[],
        cooldown=["hip flexor + hamstring stretch 3min"],
        stresses=["quads", "hamstrings", "glutes", "legs", "knees"],
    ),
    "full_body_moderate": SessionTemplate(
        key="full_body_moderate",
        primary_goal="full body balanced work",
        warmup=["5min bike easy", "world's greatest stretch 2x/side"],
        main_lifts=["goblet squat", "chest-supported row"],
        accessories=["push-up variation", "single-leg RDL", "plank"],
        conditioning=["5min easy zone-2"],
        cooldown=["full body stretch 5min"],
        stresses=["legs", "upper_body", "core"],
    ),
    "full_body_recovery": SessionTemplate(
        key="full_body_recovery",
        primary_goal="active recovery; blood flow, mobility, light loading",
        warmup=["5min easy walk or bike"],
        main_lifts=[],
        accessories=[
            "bodyweight squat 2x10 slow",
            "incline push-up 2x8",
            "band row 2x12",
            "dead bug 2x8",
        ],
        conditioning=["15-20min zone-2 bike"],
        cooldown=["full body mobility 8-10min"],
        stresses=["general"],
    ),
    "zone2_conditioning": SessionTemplate(
        key="zone2_conditioning",
        primary_goal="aerobic base + active recovery",
        warmup=["5min easy bike"],
        main_lifts=[],
        accessories=[],
        conditioning=["30-40min zone-2 bike or incline walk"],
        cooldown=["stretch 5min"],
        stresses=["aerobic"],
    ),
    "mobility_only": SessionTemplate(
        key="mobility_only",
        primary_goal="mobility + parasympathetic recovery",
        warmup=["5min easy walk"],
        main_lifts=[],
        accessories=[],
        conditioning=[],
        cooldown=[
            "hip mobility flow 5min",
            "thoracic mobility 5min",
            "breathing drill 5min",
            "full body stretch 5min",
        ],
        stresses=["mobility"],
    ),
}


# Equipment restrictions -- crude filter; exercise_prescriber does fine-grained work.
_FULL_GYM_ONLY = {"lower", "upper_push", "upper_pull"}


def _risk_to_intensity(level: RiskLevel) -> Intensity:
    return {
        "low": "hard",
        "moderate": "moderate",
        "elevated": "light",
        "high": "recovery",
    }[level]


def _extract_pain_parts(
    factors: List[RiskFactor], threshold_severity: str = "medium"
) -> List[str]:
    """Return body parts with pain at or above a given severity bucket."""
    sev_order = {"small": 0, "medium": 1, "large": 2}
    min_sev = sev_order[threshold_severity]
    parts: List[str] = []
    for f in factors:
        if f.name.startswith("pain") and sev_order[f.severity] >= min_sev:
            parts.extend(f.tags)
    return [p.lower() for p in parts]


def _recent_heavy_groups(recent_training: list) -> List[str]:
    """Muscle-group hints from recent heavy sessions (last 48h).

    We match on session 'type' keywords. Very rough but good enough for routing.
    """
    hot: List[str] = []
    keywords = {
        "squat": ["legs", "quads"],
        "deadlift": ["legs", "hamstrings", "back"],
        "leg": ["legs"],
        "lower": ["legs"],
        "pull": ["back", "biceps", "upper_body"],
        "push": ["chest", "shoulders", "upper_body"],
        "bench": ["chest", "shoulders"],
        "press": ["shoulders", "chest"],
        "upper": ["upper_body"],
    }
    for s in recent_training:
        if s.days_ago <= 2 and (s.rpe is None or s.rpe >= 7):
            t = s.type.lower()
            for k, groups in keywords.items():
                if k in t:
                    hot.extend(groups)
    return sorted(set(hot))


def _template_conflicts(
    template: SessionTemplate,
    hot_groups: List[str],
    pain_parts: List[str],
) -> int:
    """Heuristic conflict score between a template and today's avoid-list.

    Higher = more overlap with recently-trashed muscles / painful joints.
    """
    score = 0
    for g in hot_groups:
        if g in template.stresses:
            score += 2
    for p in pain_parts:
        # crude: pain body part vs template stresses / key
        if "knee" in p and ("legs" in template.stresses or template.key == "lower"):
            score += 3
        if "back" in p and (
            "back" in template.stresses or template.key in ("lower", "upper_pull")
        ):
            score += 3
        if "shoulder" in p and (
            "shoulders" in template.stresses or template.key == "upper_push"
        ):
            score += 3
        if "hip" in p and template.key == "lower":
            score += 3
        if "elbow" in p and template.key in ("upper_push", "upper_pull"):
            score += 2
        if "wrist" in p and template.key in ("upper_push", "upper_full"):
            score += 2
    return score


def _select_template(
    request: TodayRequest,
    intensity: Intensity,
    factors: List[RiskFactor],
) -> SessionTemplate:
    hot = _recent_heavy_groups(request.recent_training)
    pain = _extract_pain_parts(factors, threshold_severity="small")

    # 1) Extreme-downgrade modalities
    if intensity == "recovery":
        # Prefer zone-2 if legs aren't beat up, else full-body recovery, else mobility.
        candidates = [
            _TEMPLATES["full_body_recovery"],
            _TEMPLATES["zone2_conditioning"],
            _TEMPLATES["mobility_only"],
        ]
    elif intensity == "light":
        # Light day: default to upper_full (leg-sparing) unless upper is overworked.
        upper_hot = {"chest", "shoulders", "upper_body", "back"} & set(hot)
        if (
            upper_hot
            and "legs" not in hot
            and not any("knee" in p or "hip" in p for p in pain)
        ):
            candidates = [
                _TEMPLATES["full_body_moderate"],
                _TEMPLATES["lower"],
                _TEMPLATES["upper_full"],
            ]
        else:
            candidates = [
                _TEMPLATES["upper_full"],
                _TEMPLATES["full_body_moderate"],
                _TEMPLATES["upper_pull"],
                _TEMPLATES["upper_push"],
            ]
    else:
        # moderate or hard: full strength menu
        candidates = [
            _TEMPLATES["upper_push"],
            _TEMPLATES["upper_pull"],
            _TEMPLATES["upper_full"],
            _TEMPLATES["lower"],
            _TEMPLATES["full_body_moderate"],
        ]

    # 2) Drop templates that require equipment the user doesn't have.
    if request.equipment in ("home_basic", "bodyweight"):
        candidates = [
            c for c in candidates if c.key not in _FULL_GYM_ONLY
        ] or candidates

    # 3) Rank by conflict score (lower is better); break ties by list order.
    ranked = sorted(
        enumerate(candidates),
        key=lambda ix: (_template_conflicts(ix[1], hot, pain), ix[0]),
    )
    return ranked[0][1]


def _why_today(
    factors: List[RiskFactor], intensity: Intensity, template: SessionTemplate
) -> str:
    severity_rank = {"large": 2, "medium": 1, "small": 0}
    top = sorted(factors, key=lambda f: -severity_rank[f.severity])[:3]
    if not top:
        bits = ["all recovery signals in range"]
    else:
        bits = [f.note for f in top]
    head = " + ".join(bits)
    return (
        f"{head} -> {intensity} {template.primary_goal}. "
        "Training is preferred over rest unless signals stack severely."
    )


def plan(
    request: TodayRequest,
    risk_level: RiskLevel,
    factors: List[RiskFactor],
) -> Tuple[TodayDecision, SessionTemplate]:
    """Map risk -> intensity, pick a template, emit TodayDecision.

    Rules:
      - low -> hard; moderate -> moderate; elevated -> light; high -> recovery.
      - Only collapse to "rest" when the risk stack is extreme (3+ large factors).
    """
    intensity: Intensity = _risk_to_intensity(risk_level)

    # Only collapse to full rest on severe stacking.
    if risk_level == "high" and is_extreme_stack(factors):
        intensity = "rest"

    if intensity == "rest":
        template = _TEMPLATES["mobility_only"]
        primary_goal = "mandatory rest; parasympathetic recovery"
    else:
        template = _select_template(request, intensity, factors)
        # Merge goal_priority into primary_goal_today for clarity.
        primary_goal = template.primary_goal
        if request.goal_priority and request.goal_priority != "health+strength":
            primary_goal = f"{template.primary_goal} ({request.goal_priority})"

    decision = TodayDecision(
        risk_level=risk_level,
        primary_goal_today=primary_goal,
        recommended_intensity=intensity,
        why_today=_why_today(factors, intensity, template),
    )
    return decision, template


def get_template(key: str) -> Optional[SessionTemplate]:
    """Public accessor, kept for testability."""
    return _TEMPLATES.get(key)
