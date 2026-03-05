"""Data validation module using Pydantic schemas."""

from .schemas import (
    BodyState,
    WorkoutRecommendation,
    TrainingEntry,
    UserProfile,
    RecommendationRequest,
    RecommendationResponse,
    FeedbackRequest,
    HealthCheckResponse,
    WorkoutType,
    Intensity,
    FitnessGoal,
    DataSource,
    validate_body_state,
    validate_training_entry,
)

__all__ = [
    "BodyState",
    "WorkoutRecommendation",
    "TrainingEntry",
    "UserProfile",
    "RecommendationRequest",
    "RecommendationResponse",
    "FeedbackRequest",
    "HealthCheckResponse",
    "WorkoutType",
    "Intensity",
    "FitnessGoal",
    "DataSource",
    "validate_body_state",
    "validate_training_entry",
]
