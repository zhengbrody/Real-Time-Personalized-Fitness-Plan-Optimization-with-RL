"""
Shared Feature Transform
========================

The single source of truth for turning a raw body-state record into the model
input vector.  Both the offline training/benchmark path and the online serving
path import ``transform`` from here — that is what keeps training and serving
from drifting apart.

Design rules (these are what the parity test enforces):

1. **Pure function.**  ``transform`` depends only on its argument.  No clocks,
   no globals, no RNG, no I/O.  The same dict always yields the same vector.
2. **Fixed order.**  ``FEATURE_NAMES`` defines the column order; the vector is
   built by iterating that list, never by iterating a dict.
3. **Explicit defaults.**  Every field has a documented fallback, so a partial
   record produces a valid vector plus a missing-indicator flag rather than a
   crash or a silent NaN.
4. **Bounded output.**  Every feature is normalised into roughly [-1, 1] (a few
   ratios can exceed it slightly), which is what the linear/neural posteriors
   assume.

If you add a feature, add it to ``FEATURE_NAMES`` and to ``transform`` in the
same commit.  ``test_feature_parity.py`` fails if the two disagree.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

RECOVERY_FEATURES: List[str] = [
    "readiness",  # Oura-style readiness score, 0-100 -> [0, 1]
    "sleep_score",  # sleep quality score, 0-100 -> [0, 1]
    "sleep_hours",  # hours slept, centred on 7.5h
    "sleep_debt",  # cumulative 7d shortfall vs 7.5h target, hours
    "hrv",  # last night's HRV (ms), centred on population mean
    "hrv_7d_mean",  # 7-day rolling mean HRV
    "hrv_z",  # z-score of today's HRV vs the user's own 28d baseline
    "hrv_trend",  # 7d mean minus 28d mean, normalised
    "resting_hr_dev",  # deviation from the user's own resting-HR baseline
]

LOAD_FEATURES: List[str] = [
    "fatigue",  # self-reported 1-10 -> [0, 1]
    "activity_score",  # yesterday's activity score, 0-100 -> [0, 1]
    "acwr",  # acute:chronic workload ratio, centred on 1.0
    "load_7d",  # 7-day training load sum, normalised
    "load_28d",  # 28-day training load mean, normalised
    "days_since_training",  # capped at 7
    "consecutive_hard_days",  # capped at 5
]

CONSISTENCY_FEATURES: List[str] = [
    "completion_rate_7d",  # fraction of prescribed sessions completed
    "completion_rate_28d",
    "streak",  # consecutive active days, capped at 14
    "sessions_7d",  # sessions in the last 7 days, capped at 7
    "days_active_28d",  # active days in last 28, -> [0, 1]
]

TEMPORAL_FEATURES: List[str] = [
    "dow_sin",  # day-of-week encoded on the unit circle so that
    "dow_cos",  # Sunday and Monday are adjacent, not 6 apart
    "is_weekend",
    "week_phase",  # position within the mesocycle, [0, 1]
]

PROFILE_FEATURES: List[str] = [
    "training_age",  # years of consistent training, capped at 10
    "age",  # chronological age, normalised around 35
    "goal_strength",  # one-hot-ish goal encoding
    "goal_endurance",
    "intensity_tolerance",  # self-reported historical tolerance, [0, 1]
    "bodyweight",  # kg, centred on 75
]

MISSING_FEATURES: List[str] = [
    "hrv_missing",
    "sleep_missing",
]

BIAS_FEATURES: List[str] = ["bias"]

FEATURE_NAMES: List[str] = (
    RECOVERY_FEATURES
    + LOAD_FEATURES
    + CONSISTENCY_FEATURES
    + TEMPORAL_FEATURES
    + PROFILE_FEATURES
    + MISSING_FEATURES
    + BIAS_FEATURES
)

FEATURE_DIM: int = len(FEATURE_NAMES)

# Population baselines used for centring.  These are constants, not fitted
# statistics, so that the transform stays pure and serving needs no state.
_HRV_BASELINE_MS = 50.0
_HRV_SCALE_MS = 25.0
_RHR_BASELINE_BPM = 60.0
_RHR_SCALE_BPM = 15.0
_SLEEP_TARGET_H = 7.5
_AGE_CENTRE = 35.0
_AGE_SCALE = 25.0
_BODYWEIGHT_CENTRE_KG = 75.0
_BODYWEIGHT_SCALE_KG = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num(state: Mapping[str, Any], key: str, default: float) -> float:
    """Read a numeric field, treating None/NaN/non-numeric as absent."""
    value = state.get(key, None)
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(max(value, lo), hi))


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def transform(state: Mapping[str, Any]) -> np.ndarray:
    """
    Turn one raw body-state record into the model input vector.

    Parameters
    ----------
    state
        Mapping of raw field names to values.  Missing fields fall back to the
        documented population defaults and raise the corresponding
        ``*_missing`` indicator.

    Returns
    -------
    np.ndarray
        ``float64`` vector of length :data:`FEATURE_DIM`, ordered exactly as
        :data:`FEATURE_NAMES`.
    """
    hrv_present = state.get("hrv", None) is not None
    sleep_present = state.get("sleep_score", None) is not None

    readiness = _num(state, "readiness_score", 50.0)
    sleep_score = _num(state, "sleep_score", 70.0)
    sleep_hours = _num(state, "sleep_duration_hours", _SLEEP_TARGET_H)
    sleep_debt = _num(state, "sleep_debt_7d", 0.0)
    hrv = _num(state, "hrv", _HRV_BASELINE_MS)
    hrv_7d = _num(state, "hrv_7d_mean", hrv)
    hrv_28d = _num(state, "hrv_28d_mean", hrv_7d)
    hrv_std = _num(state, "hrv_28d_std", _HRV_SCALE_MS / 2.0)
    resting_hr = _num(state, "resting_hr", _RHR_BASELINE_BPM)
    resting_hr_base = _num(state, "resting_hr_baseline", _RHR_BASELINE_BPM)

    fatigue = _num(state, "fatigue", 5.0)
    activity = _num(state, "activity_score", 50.0)
    acwr = _num(state, "acwr", 1.0)
    load_7d = _num(state, "load_7d", 0.0)
    load_28d = _num(state, "load_28d_mean", 0.0)
    days_since = _num(state, "days_since_training", 1.0)
    consec_hard = _num(state, "consecutive_hard_days", 0.0)

    completion_7d = _num(state, "completion_rate_7d", 0.7)
    completion_28d = _num(state, "completion_rate_28d", 0.7)
    streak = _num(state, "streak", 0.0)
    sessions_7d = _num(state, "sessions_7d", 3.0)
    days_active_28d = _num(state, "days_active_28d", 14.0)

    day_of_week = int(_num(state, "day_of_week", 0.0)) % 7
    week_phase = _num(state, "week_phase", (day_of_week % 7) / 7.0)

    training_age = _num(state, "training_age_years", 3.0)
    age = _num(state, "age", _AGE_CENTRE)
    goal = str(state.get("goal", "general") or "general").lower()
    intensity_tolerance = _num(state, "intensity_tolerance", 0.5)
    bodyweight = _num(state, "bodyweight_kg", _BODYWEIGHT_CENTRE_KG)

    # hrv_std can legitimately be 0 for a brand-new user with one reading.
    hrv_z = (hrv - hrv_28d) / hrv_std if hrv_std > 1e-6 else 0.0

    values: Dict[str, float] = {
        # -- recovery ------------------------------------------------------
        "readiness": _clip(readiness / 100.0, 0.0, 1.0),
        "sleep_score": _clip(sleep_score / 100.0, 0.0, 1.0),
        "sleep_hours": _clip((sleep_hours - _SLEEP_TARGET_H) / 3.0),
        "sleep_debt": _clip(sleep_debt / 14.0),
        "hrv": _clip((hrv - _HRV_BASELINE_MS) / _HRV_SCALE_MS),
        "hrv_7d_mean": _clip((hrv_7d - _HRV_BASELINE_MS) / _HRV_SCALE_MS),
        "hrv_z": _clip(hrv_z / 3.0),
        "hrv_trend": _clip((hrv_7d - hrv_28d) / _HRV_SCALE_MS),
        "resting_hr_dev": _clip((resting_hr - resting_hr_base) / _RHR_SCALE_BPM),
        # -- load ----------------------------------------------------------
        "fatigue": _clip(fatigue / 10.0, 0.0, 1.0),
        "activity_score": _clip(activity / 100.0, 0.0, 1.0),
        "acwr": _clip((acwr - 1.0) / 1.0),
        "load_7d": _clip(load_7d / 1000.0, 0.0, 1.0),
        "load_28d": _clip(load_28d / 1000.0, 0.0, 1.0),
        "days_since_training": _clip(days_since / 7.0, 0.0, 1.0),
        "consecutive_hard_days": _clip(consec_hard / 5.0, 0.0, 1.0),
        # -- consistency ---------------------------------------------------
        "completion_rate_7d": _clip(completion_7d, 0.0, 1.0),
        "completion_rate_28d": _clip(completion_28d, 0.0, 1.0),
        "streak": _clip(streak / 14.0, 0.0, 1.0),
        "sessions_7d": _clip(sessions_7d / 7.0, 0.0, 1.0),
        "days_active_28d": _clip(days_active_28d / 28.0, 0.0, 1.0),
        # -- temporal ------------------------------------------------------
        "dow_sin": math.sin(2.0 * math.pi * day_of_week / 7.0),
        "dow_cos": math.cos(2.0 * math.pi * day_of_week / 7.0),
        "is_weekend": 1.0 if day_of_week >= 5 else 0.0,
        "week_phase": _clip(week_phase, 0.0, 1.0),
        # -- profile -------------------------------------------------------
        "training_age": _clip(training_age / 10.0, 0.0, 1.0),
        "age": _clip((age - _AGE_CENTRE) / _AGE_SCALE),
        "goal_strength": 1.0 if goal == "strength" else 0.0,
        "goal_endurance": 1.0 if goal == "endurance" else 0.0,
        "intensity_tolerance": _clip(intensity_tolerance, 0.0, 1.0),
        "bodyweight": _clip(
            (bodyweight - _BODYWEIGHT_CENTRE_KG) / _BODYWEIGHT_SCALE_KG
        ),
        # -- missing indicators --------------------------------------------
        "hrv_missing": 0.0 if hrv_present else 1.0,
        "sleep_missing": 0.0 if sleep_present else 1.0,
        # -- bias ----------------------------------------------------------
        "bias": 1.0,
    }

    # Build by iterating FEATURE_NAMES so order can never drift from the schema.
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)


def transform_many(states: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Vectorised convenience wrapper: returns an ``(n, FEATURE_DIM)`` matrix."""
    if not states:
        return np.zeros((0, FEATURE_DIM), dtype=np.float64)
    return np.vstack([transform(s) for s in states])


def describe(vector: np.ndarray) -> Dict[str, float]:
    """Map a feature vector back to ``{name: value}`` — for debugging and logs."""
    if vector.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM} features, got {vector.shape[-1]}")
    return {name: float(vector[i]) for i, name in enumerate(FEATURE_NAMES)}
