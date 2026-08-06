"""End-to-end tests for the dual-layer recommendation pipeline (Stage-2).

Each test constructs a TodayRequest directly via pydantic and asserts on the
returned Recommendation shape.
"""

import pytest

from src.recommendation import recommend_today
from src.validation.schemas import (
    ManualCheckIn,
    RecentSession,
    Recommendation,
    TodayRequest,
    WearableData,
)


def _acceptance_request() -> TodayRequest:
    return TodayRequest(
        goal_priority="health+strength",
        equipment="full_gym",
        time_budget_min=75,
        wearable=WearableData(
            readiness_score=68,
            sleep_score=61,
            sleep_hours=6.1,
            hrv=41,
            resting_hr=63,
            activity_score=54,
        ),
        manual=ManualCheckIn(
            fatigue=7,
            motivation=6,
            pain={"left_knee": 2},
            soreness={"legs": 6, "upper_body": 2},
        ),
        recent_training=[
            RecentSession(days_ago=1, type="heavy squat", rpe=9),
            RecentSession(days_ago=2, type="pull day", rpe=8),
            RecentSession(days_ago=3, type="rest"),
        ],
    )


class TestAcceptanceSample:
    def test_acceptance_sample_matches_contract(self):
        rec = recommend_today(_acceptance_request())

        assert isinstance(rec, Recommendation)
        assert rec.today_decision.risk_level == "elevated"
        assert rec.today_decision.recommended_intensity == "light"

        # No heavy bilateral squat / deadlift in the main lifts.
        main_lower = [m.lower() for m in rec.session_plan.main_lifts]
        for bad in ("back squat", "front squat", "conventional deadlift"):
            assert bad not in main_lower

        # Forbidden list should mention the knee or heavy bilateral lower work.
        forbidden_text = " | ".join(rec.forbidden_or_not_recommended).lower()
        assert (
            "knee" in forbidden_text
            or "squat" in forbidden_text
            or "lower" in forbidden_text
        )

        assert rec.override_option
        assert rec.stop_conditions


class TestGracefulDegradation:
    def test_only_fatigue_still_returns_valid_recommendation(self):
        req = TodayRequest(manual=ManualCheckIn(fatigue=5))
        rec = recommend_today(req)

        assert isinstance(rec, Recommendation)
        assert rec.today_decision.recommended_intensity in (
            "rest",
            "recovery",
            "light",
            "moderate",
            "hard",
        )

        # Wearable data gaps should be flagged.
        gaps = " | ".join(rec.data_gaps).lower()
        for missing in ("readiness", "sleep", "hrv", "resting_hr"):
            assert missing in gaps


class TestOverridePreference:
    def _base(self) -> TodayRequest:
        return TodayRequest(
            wearable=WearableData(
                readiness_score=60, sleep_hours=7.0, hrv=45, resting_hr=62
            ),
            manual=ManualCheckIn(fatigue=5),
        )

    def test_override_preferences_produce_different_guidance(self):
        base = self._base()

        follow_req = base.model_copy(
            update={"override_preference": "follow_recommendation"}
        )
        heavier_req = base.model_copy(
            update={"override_preference": "go_heavier_if_possible"}
        )
        lighter_req = base.model_copy(update={"override_preference": "go_lighter"})

        follow = recommend_today(follow_req).override_option
        heavier = recommend_today(heavier_req).override_option
        lighter = recommend_today(lighter_req).override_option

        # All three strings should be non-empty and at least two must differ.
        assert follow and heavier and lighter
        assert len({follow, heavier, lighter}) >= 2


class TestPainLogic:
    def test_high_knee_pain_forbids_heavy_knee_loading(self):
        req = TodayRequest(
            wearable=WearableData(
                readiness_score=75,
                sleep_hours=8.0,
                hrv=60,
                resting_hr=58,
                sleep_score=85,
                activity_score=70,
            ),
            manual=ManualCheckIn(fatigue=3, pain={"left_knee": 7}),
        )
        rec = recommend_today(req)

        forbidden_text = " | ".join(rec.forbidden_or_not_recommended).lower()
        assert "knee" in forbidden_text

        main_lower = [m.lower() for m in rec.session_plan.main_lifts]
        for bad in ("back squat", "front squat", "bulgarian split squat"):
            assert bad not in main_lower


class TestRecentTrainingLogic:
    def test_two_consecutive_hard_days_raise_risk(self):
        base_wearable = WearableData(
            readiness_score=80,
            sleep_hours=8.0,
            hrv=60,
            resting_hr=58,
            sleep_score=85,
            activity_score=70,
        )
        base_manual = ManualCheckIn(fatigue=3)

        fresh_req = TodayRequest(wearable=base_wearable, manual=base_manual)
        loaded_req = TodayRequest(
            wearable=base_wearable,
            manual=base_manual,
            recent_training=[
                RecentSession(days_ago=1, type="lower heavy", rpe=9),
                RecentSession(days_ago=2, type="upper heavy", rpe=9),
            ],
        )

        risk_order = {"low": 0, "moderate": 1, "elevated": 2, "high": 3}
        fresh_level = recommend_today(fresh_req).today_decision.risk_level
        loaded_level = recommend_today(loaded_req).today_decision.risk_level

        assert risk_order[loaded_level] >= risk_order[fresh_level] + 1
