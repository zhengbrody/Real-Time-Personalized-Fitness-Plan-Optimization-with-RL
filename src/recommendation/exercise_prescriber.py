"""
Exercise Prescriber

Expand a chosen session template's named exercises into ExerciseRx rows with
sets/reps/target_rpe/load_guidance scaled by the session intensity and
substitution strings that honor pain and equipment constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.validation.schemas import ExerciseRx, TodayDecision, TodayRequest
from .session_planner import SessionTemplate

# ---------------------------------------------------------------------------
# Catalog: one primary + 1-2 alternates per movement category. Not exhaustive.
# ---------------------------------------------------------------------------


@dataclass
class ExerciseInfo:
    name: str
    category: str  # squat, hinge, horizontal_push, vertical_push, horizontal_pull,
    # vertical_pull, lunge, carry, core, conditioning, accessory
    # body parts this exercise loads (used for pain-based substitution).
    loads: List[str]


_CATALOG: Dict[str, ExerciseInfo] = {
    "back squat": ExerciseInfo("back squat", "squat", ["knees", "legs", "lower_back"]),
    "front squat": ExerciseInfo("front squat", "squat", ["knees", "legs"]),
    "goblet squat": ExerciseInfo("goblet squat", "squat", ["legs"]),
    "bulgarian split squat": ExerciseInfo(
        "bulgarian split squat", "lunge", ["knees", "legs"]
    ),
    "romanian deadlift": ExerciseInfo(
        "romanian deadlift", "hinge", ["hamstrings", "lower_back"]
    ),
    "single-leg RDL": ExerciseInfo("single-leg RDL", "hinge", ["hamstrings"]),
    "bench press": ExerciseInfo(
        "bench press", "horizontal_push", ["chest", "shoulders", "triceps"]
    ),
    "incline db press": ExerciseInfo(
        "incline db press", "horizontal_push", ["chest", "shoulders"]
    ),
    "push-up variation": ExerciseInfo(
        "push-up variation", "horizontal_push", ["chest", "shoulders"]
    ),
    "dips or push-ups": ExerciseInfo(
        "dips or push-ups", "horizontal_push", ["chest", "triceps"]
    ),
    "overhead press": ExerciseInfo(
        "overhead press", "vertical_push", ["shoulders", "triceps"]
    ),
    "weighted pull-up": ExerciseInfo(
        "weighted pull-up", "vertical_pull", ["back", "biceps"]
    ),
    "lat pulldown": ExerciseInfo("lat pulldown", "vertical_pull", ["back", "biceps"]),
    "barbell row": ExerciseInfo(
        "barbell row", "horizontal_pull", ["back", "lower_back", "biceps"]
    ),
    "chest-supported row": ExerciseInfo(
        "chest-supported row", "horizontal_pull", ["back", "biceps"]
    ),
    "face pull": ExerciseInfo("face pull", "accessory", ["rear_delts"]),
    "triceps extension": ExerciseInfo("triceps extension", "accessory", ["triceps"]),
    "biceps curl": ExerciseInfo("biceps curl", "accessory", ["biceps"]),
    "hammer curl": ExerciseInfo("hammer curl", "accessory", ["biceps"]),
    "leg curl": ExerciseInfo("leg curl", "accessory", ["hamstrings"]),
    "standing calf raise": ExerciseInfo("standing calf raise", "accessory", ["calves"]),
    "dead bug": ExerciseInfo("dead bug", "core", ["core"]),
    "plank": ExerciseInfo("plank", "core", ["core"]),
    "bodyweight squat 2x10 slow": ExerciseInfo(
        "bodyweight squat 2x10 slow", "accessory", ["legs"]
    ),
    "incline push-up 2x8": ExerciseInfo(
        "incline push-up 2x8", "accessory", ["chest", "shoulders"]
    ),
    "band row 2x12": ExerciseInfo("band row 2x12", "accessory", ["back"]),
    "dead bug 2x8": ExerciseInfo("dead bug 2x8", "core", ["core"]),
}


# ---------------------------------------------------------------------------
# Intensity profiles for sets/reps/RPE/load guidance.
# ---------------------------------------------------------------------------


def _profile_main(intensity: str) -> Dict[str, str | int]:
    return {
        "hard": {"sets": 5, "reps": "5", "rpe": "RPE 8", "load": "75-85% 1RM"},
        "moderate": {"sets": 4, "reps": "6-8", "rpe": "RPE 7", "load": "65-75% 1RM"},
        "light": {"sets": 3, "reps": "8-10", "rpe": "RPE 6", "load": "55-65% 1RM"},
        "recovery": {
            "sets": 2,
            "reps": "10-12",
            "rpe": "technique only",
            "load": "bodyweight or light DB",
        },
        "rest": {"sets": 0, "reps": "-", "rpe": "-", "load": "no loading"},
    }[intensity]


def _profile_accessory(intensity: str) -> Dict[str, str | int]:
    return {
        "hard": {"sets": 3, "reps": "8-10", "rpe": "RPE 8", "load": "moderate-heavy"},
        "moderate": {
            "sets": 3,
            "reps": "10-12",
            "rpe": "RPE 7",
            "load": "moderate",
        },
        "light": {"sets": 3, "reps": "10-12", "rpe": "RPE 6", "load": "light-moderate"},
        "recovery": {
            "sets": 2,
            "reps": "10-15",
            "rpe": "RPE 5",
            "load": "light / bodyweight",
        },
        "rest": {"sets": 0, "reps": "-", "rpe": "-", "load": "no loading"},
    }[intensity]


# ---------------------------------------------------------------------------
# Pain-aware substitution logic.
# ---------------------------------------------------------------------------


def _collect_pain(request: TodayRequest) -> Dict[str, int]:
    return {
        k.lower(): v for k, v in (request.manual.pain or {}).items() if v is not None
    }


_SUB_RULES: Dict[str, List[Dict]] = {
    # match key -> list of (when_pain_contains, threshold, alt)
    "back squat": [
        {"pain": "knee", "threshold": 3, "alt": "goblet squat (knee-friendlier)"},
        {"pain": "back", "threshold": 3, "alt": "front squat or belt squat"},
    ],
    "front squat": [
        {"pain": "knee", "threshold": 3, "alt": "goblet squat"},
    ],
    "bulgarian split squat": [
        {"pain": "knee", "threshold": 3, "alt": "step-up with shorter range"},
    ],
    "romanian deadlift": [
        {"pain": "back", "threshold": 3, "alt": "single-leg RDL (lighter load)"},
    ],
    "barbell row": [
        {"pain": "back", "threshold": 3, "alt": "chest-supported row"},
    ],
    "bench press": [
        {"pain": "shoulder", "threshold": 3, "alt": "neutral-grip db press"},
        {"pain": "wrist", "threshold": 3, "alt": "db floor press"},
    ],
    "overhead press": [
        {"pain": "shoulder", "threshold": 3, "alt": "landmine press"},
    ],
    "weighted pull-up": [
        {"pain": "elbow", "threshold": 3, "alt": "neutral-grip lat pulldown"},
    ],
    "dips or push-ups": [
        {"pain": "shoulder", "threshold": 3, "alt": "incline push-up"},
        {"pain": "elbow", "threshold": 3, "alt": "incline push-up"},
    ],
}


def _substitution_for(name: str, pain: Dict[str, int]) -> Optional[str]:
    rules = _SUB_RULES.get(name, [])
    for r in rules:
        for part, level in pain.items():
            if r["pain"] in part and level >= r["threshold"]:
                return f"{name} -> {r['alt']} if {part} pain >= {r['threshold']}"
    # Fallback: if this exercise hits a painful area without an explicit rule,
    # surface a generic substitution hint.
    info = _CATALOG.get(name)
    if info:
        for part, level in pain.items():
            if level >= 4 and any(part.split("_")[-1] in l for l in info.loads):
                return f"swap {name} for a lighter/isolation variant (pain in {part})"
    return None


def _equipment_guard(name: str, equipment: str) -> Optional[str]:
    if equipment in ("home_basic", "bodyweight"):
        # For barbell-centric movements, suggest a swap.
        barbell_moves = {
            "back squat": "goblet squat or bodyweight squat",
            "front squat": "goblet squat",
            "romanian deadlift": "single-leg RDL with db/backpack",
            "bench press": "push-up variation or floor press with db",
            "barbell row": "bent-over db row or band row",
            "overhead press": "db shoulder press or pike push-up",
            "weighted pull-up": "bodyweight pull-up or inverted row",
        }
        if name in barbell_moves:
            return f"{name} -> {barbell_moves[name]} (no full gym today)"
    return None


def _build_rx(
    name: str, slot: str, intensity: str, request: TodayRequest
) -> ExerciseRx:
    profile = (
        _profile_main(intensity) if slot == "main" else _profile_accessory(intensity)
    )
    pain = _collect_pain(request)

    # Equipment guard wins over pain guard (you physically cannot do the lift);
    # then fall back to pain-based substitution hint.
    sub = _equipment_guard(name, request.equipment) or _substitution_for(name, pain)

    return ExerciseRx(
        name=name,
        sets=int(profile["sets"]),
        reps=str(profile["reps"]),
        target_rpe=str(profile["rpe"]),
        load_guidance=str(profile["load"]),
        substitution_if_needed=sub,
    )


def prescribe(
    template: SessionTemplate,
    today_decision: TodayDecision,
    request: TodayRequest,
) -> List[ExerciseRx]:
    intensity = today_decision.recommended_intensity
    rxs: List[ExerciseRx] = []
    for name in template.main_lifts:
        rxs.append(_build_rx(name, "main", intensity, request))
    for name in template.accessories:
        rxs.append(_build_rx(name, "accessory", intensity, request))
    return rxs
