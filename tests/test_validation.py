"""Tests for the validation schemas module."""

import pytest
from src.validation.schemas import (
    BodyState,
    WorkoutRecommendation,
    RecommendationRequest,
    HealthCheckResponse,
)
from pydantic import ValidationError


class TestBodyState:
    """Tests for the BodyState schema."""

    def test_valid_body_state(self):
        state = BodyState(
            readiness_score=75,
            sleep_score=80,
            hrv=55,
            resting_hr=60,
            activity_score=70,
            fatigue=3,
        )
        assert state.readiness_score == 75

    def test_readiness_score_range(self):
        with pytest.raises(ValidationError):
            BodyState(
                readiness_score=150,  # Out of range
                sleep_score=80,
                hrv=55,
                resting_hr=60,
                activity_score=70,
                fatigue=3,
            )

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            BodyState(
                readiness_score=75,
                # Missing required fields
            )

    def test_fatigue_range(self):
        """Test fatigue is in valid range."""
        with pytest.raises(ValidationError):
            BodyState(
                readiness_score=75,
                sleep_score=80,
                hrv=55,
                resting_hr=60,
                activity_score=70,
                fatigue=15,  # Out of range (max 10)
            )


class TestHealthCheckResponse:
    """Tests for the HealthCheckResponse schema."""

    def test_valid_health_check(self):
        response = HealthCheckResponse(
            status="healthy",
        )
        assert response.status == "healthy"

    def test_health_check_with_services(self):
        response = HealthCheckResponse(
            status="healthy",
            services={"api": "healthy", "database": "degraded"},
        )
        assert response.status == "healthy"
        assert response.services["api"] == "healthy"
