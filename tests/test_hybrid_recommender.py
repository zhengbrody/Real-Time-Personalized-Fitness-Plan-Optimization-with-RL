"""Tests for the hybrid recommender module.

Stage-2: the engine now returns the dual-layer `Recommendation` shape.
These tests cover both the new module-level `recommend(TodayRequest)` entry
point and the legacy-dict shim on `HybridRecommender.recommend`, which
builds its dict from the same `Recommendation`.
"""

import pytest
from src.recommendation.hybrid_recommender import HybridRecommender, recommend
from src.validation.schemas import (
    ManualCheckIn,
    RecentSession,
    Recommendation,
    TodayRequest,
    WearableData,
)


def _req(**overrides) -> TodayRequest:
    """Small helper to build a TodayRequest with sane defaults."""
    wearable = WearableData(**overrides.pop("wearable", {}))
    manual = ManualCheckIn(**overrides.pop("manual", {}))
    recent = overrides.pop("recent_training", [])
    return TodayRequest(
        wearable=wearable, manual=manual, recent_training=recent, **overrides
    )


class TestHybridRecommenderLegacyShim:
    """Cover the legacy dict path on `HybridRecommender.recommend`.

    The shim now proxies to the new dual-layer engine and maps its output
    back to the old dict shape, so these tests assert the mapping works.
    """

    def test_initialization(self):
        recommender = HybridRecommender()
        assert recommender.action_space is not None
        assert recommender.action_space.get_action_count() == 18
        assert recommender.bandit is not None

    def test_initialization_without_rl(self):
        recommender = HybridRecommender(use_rl=False)
        assert recommender.bandit is None

    def test_recommend_returns_dict(self):
        recommender = HybridRecommender()
        state = {
            "readiness_score": 75,
            "hrv": 50,
            "resting_hr": 60,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": {"legs": 2},
            "activity_score": 70,
        }
        result = recommender.recommend(state)
        assert isinstance(result, dict)
        for k in (
            "action_id",
            "workout_type",
            "intensity",
            "duration_minutes",
            "description",
            "safety_check",
            "rationale",
            "recommendation",
        ):
            assert k in result

    def test_recommend_low_readiness_is_conservative(self):
        recommender = HybridRecommender()
        state = {
            "readiness_score": 20,
            "hrv": 20,
            "resting_hr": 85,
            "sleep_duration_hours": 3,
            "fatigue": 9,
            "soreness": {"legs": 8},
            "activity_score": 10,
        }
        result = recommender.recommend(state)
        # Legacy intensity should collapse to LOW/NONE with such a profile.
        assert result["intensity"] in ("LOW", "NONE")
        assert result["workout_type"] in ("REST", "RECOVERY")

    def test_recommend_high_readiness_permits_strength(self):
        recommender = HybridRecommender()
        state = {
            "readiness_score": 95,
            "hrv": 80,
            "resting_hr": 50,
            "sleep_duration_hours": 9,
            "fatigue": 1,
            "soreness": {"legs": 1},
            "activity_score": 95,
        }
        result = recommender.recommend(state)
        assert result["workout_type"] == "STRENGTH"
        assert result["intensity"] in ("MEDIUM", "HIGH")

    def test_update_with_bandit_does_not_raise(self):
        recommender = HybridRecommender(use_rl=True)
        state = {
            "readiness_score": 75,
            "hrv": 50,
            "resting_hr": 60,
            "sleep_duration_hours": 8,
            "fatigue": 3,
            "soreness": {"legs": 2},
            "activity_score": 70,
        }
        result = recommender.recommend(state)
        recommender.update(result["action_id"], state, 0.8)
        assert recommender.bandit.action_counts[result["action_id"]] == 1

    def test_update_without_bandit_is_noop(self):
        recommender = HybridRecommender(use_rl=False)
        assert recommender.bandit is None
        recommender.update(0, {}, 0.5)


class TestHybridRecommenderDualLayer:
    """Scenarios ported from the old single-action tests -- now asserting the
    new dual-layer `Recommendation` shape produced by `recommend(TodayRequest)`.
    """

    def test_low_readiness_downshifts_intensity(self):
        """Old: low readiness -> REST. New: low readiness stacked with other
        mild issues raises risk to at least elevated, yielding a light/recovery/rest session.
        """
        req = _req(
            wearable={
                "readiness_score": 25,
                "sleep_hours": 5.0,
                "hrv": 28,
                "resting_hr": 78,
            },
            manual={"fatigue": 7},
        )
        rec = recommend(req)
        assert isinstance(rec, Recommendation)
        assert rec.today_decision.risk_level in ("elevated", "high")
        assert rec.today_decision.recommended_intensity in ("light", "recovery", "rest")

    def test_low_sleep_downshifts_intensity(self):
        """Old: sleep_hours < 5 -> REST. New: severe sleep deficit stacks with
        low readiness to yield elevated/high risk and a light-or-easier session."""
        req = _req(
            wearable={
                "readiness_score": 45,
                "sleep_hours": 4.0,
                "hrv": 35,
                "resting_hr": 65,
            },
            manual={"fatigue": 6},
        )
        rec = recommend(req)
        assert rec.today_decision.risk_level in ("elevated", "high")
        assert rec.today_decision.recommended_intensity in ("light", "recovery", "rest")

    def test_high_fatigue_avoids_hard_session(self):
        """Old: fatigue >= 7 -> RECOVERY. New: severe fatigue alone may only
        land in 'moderate' risk bucket, but the session never goes to 'hard'."""
        req = _req(
            wearable={
                "readiness_score": 60,
                "sleep_hours": 7,
                "hrv": 50,
                "resting_hr": 60,
            },
            manual={"fatigue": 9},
        )
        rec = recommend(req)
        assert rec.today_decision.recommended_intensity in (
            "moderate",
            "light",
            "recovery",
            "rest",
        )
        assert rec.today_decision.recommended_intensity != "hard"

    def test_well_recovered_gets_real_session(self):
        """Old: long rest -> MEDIUM. New: recovered athlete gets moderate/hard."""
        req = _req(
            wearable={
                "readiness_score": 80,
                "sleep_hours": 8,
                "hrv": 60,
                "resting_hr": 55,
                "activity_score": 70,
                "sleep_score": 85,
            },
            manual={"fatigue": 4},
        )
        rec = recommend(req)
        assert rec.today_decision.recommended_intensity in ("moderate", "hard")
        # Recovered day: there should be actual prescribed work, not an empty plan.
        assert len(rec.exercise_prescription) >= 1

    def test_rationale_rest_like_day_references_factor(self):
        """why_today should name at least one triggering signal on a low-readiness day."""
        req = _req(
            wearable={
                "readiness_score": 30,
                "sleep_hours": 4,
                "hrv": 40,
                "resting_hr": 60,
            },
            manual={"fatigue": 3},
        )
        rec = recommend(req)
        text = rec.today_decision.why_today.lower()
        assert text
        assert any(k in text for k in ("readiness", "sleep", "hrv", "fatigue"))

    def test_rationale_light_day_references_factor(self):
        """why_today on a mid-range day should still cite a concrete factor."""
        req = _req(
            wearable={
                "readiness_score": 55,
                "sleep_hours": 6.5,
                "hrv": 45,
                "resting_hr": 62,
            },
            manual={"fatigue": 5},
        )
        rec = recommend(req)
        text = rec.today_decision.why_today.lower()
        assert text
        assert any(k in text for k in ("readiness", "sleep", "hrv", "fatigue", "range"))

    def test_rationale_moderate_or_hard_day_non_empty(self):
        """why_today is a non-empty string even when all signals are in range."""
        req = _req(
            wearable={
                "readiness_score": 85,
                "sleep_hours": 8.5,
                "hrv": 65,
                "resting_hr": 55,
                "sleep_score": 90,
                "activity_score": 80,
            },
            manual={"fatigue": 2},
        )
        rec = recommend(req)
        assert rec.today_decision.why_today
        # On a clean day, the string should at least reference the template goal
        # or state that signals are in range.
        text = rec.today_decision.why_today.lower()
        assert (
            "training" in text
            or "range" in text
            or "goal" in text
            or "strength" in text
        )


class TestEmptyAllowedActionsLegacy:
    """Port of the old 'empty allowed actions -> rest' edge case.

    The new engine never returns 'no allowed actions' (it downshifts intensity
    instead), so this scenario is retired at the engine level. We keep one
    assertion that a sufficiently extreme profile still yields the most
    conservative legacy action (REST) via the shim.
    """

    def test_extreme_profile_collapses_to_rest_via_shim(self):
        recommender = HybridRecommender(use_rl=False)
        state = {
            "readiness_score": 10,
            "sleep_duration_hours": 2,
            "hrv": 12,
            "resting_hr": 100,
            "fatigue": 10,
            "soreness": {"legs": 9, "upper_body": 9},
            "pain": {"lower_back": 8, "left_knee": 8, "right_shoulder": 8},
            "activity_score": 5,
        }
        result = recommender.recommend(state)
        assert result["workout_type"] in ("REST", "RECOVERY")
        assert result["intensity"] in ("NONE", "LOW")
