"""
Data validation schemas using Pydantic.

This module defines strict validation schemas for all data inputs to ensure
data quality and prevent errors from malformed or missing data.
"""

from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


# ============================================
# Enums for Categorical Values
# ============================================

class WorkoutType(str, Enum):
    """Valid workout types."""
    UPPER_BODY_STRENGTH = "Upper Body Strength"
    LOWER_BODY_STRENGTH = "Lower Body Strength"
    FULL_BODY_STRENGTH = "Full Body Strength"
    CARDIO_HIIT = "Cardio (HIIT)"
    CARDIO_STEADY = "Cardio (Steady State)"
    FLEXIBILITY = "Flexibility/Mobility"
    REST = "Rest Day"
    ACTIVE_RECOVERY = "Active Recovery"
    UNKNOWN = "Unknown"


class Intensity(str, Enum):
    """Workout intensity levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FitnessGoal(str, Enum):
    """User fitness goals."""
    STRENGTH = "strength"
    ENDURANCE = "endurance"
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    GENERAL_FITNESS = "general_fitness"


class DataSource(str, Enum):
    """Source of the data."""
    MANUAL_UPLOAD = "manual_upload"
    CSV_UPLOAD = "csv_upload"
    JSON_UPLOAD = "json_upload"
    API = "api"
    APPLE_HEALTH = "apple_health"
    OURA_RING = "oura_ring"


# ============================================
# Body State Schema
# ============================================

class BodyState(BaseModel):
    """
    Represents the user's current physiological state.

    All scores are validated to be within acceptable ranges.
    """

    # Oura Ring Metrics
    readiness_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Oura readiness score (0-100)"
    )
    sleep_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Oura sleep score (0-100)"
    )
    hrv: int = Field(
        ...,
        ge=0,
        le=200,
        description="Heart Rate Variability in milliseconds"
    )

    # Apple Watch Metrics
    resting_hr: int = Field(
        ...,
        ge=30,
        le=120,
        description="Resting heart rate in bpm"
    )
    activity_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Daily activity score (0-100)"
    )
    exercise_minutes: int = Field(
        default=0,
        ge=0,
        le=300,
        description="Exercise minutes today"
    )

    # Subjective Metrics
    fatigue: int = Field(
        ...,
        ge=1,
        le=10,
        description="Subjective fatigue level (1=fresh, 10=exhausted)"
    )
    mood: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Mood score (1=very bad, 10=excellent)"
    )
    stress: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Stress level (1=very low, 10=very high)"
    )

    # Optional Advanced Metrics
    body_temperature: Optional[float] = Field(
        default=None,
        ge=35.0,
        le=42.0,
        description="Body temperature in Celsius"
    )
    blood_oxygen: Optional[int] = Field(
        default=None,
        ge=80,
        le=100,
        description="Blood oxygen saturation percentage"
    )

    @validator('resting_hr')
    def validate_resting_hr(cls, v, values):
        """Validate resting heart rate is reasonable."""
        if v < 40:
            raise ValueError("Resting HR below 40 is unusual. Please verify.")
        if v > 100:
            raise ValueError("Resting HR above 100 may indicate issues. Consult a doctor.")
        return v

    @validator('hrv')
    def validate_hrv(cls, v):
        """Validate HRV is in reasonable range."""
        if v < 10:
            raise ValueError("HRV below 10ms is very low. Please verify data.")
        if v > 150:
            raise ValueError("HRV above 150ms is unusually high. Please verify.")
        return v

    @root_validator
    def validate_readiness_vs_fatigue(cls, values):
        """Check consistency between readiness and fatigue."""
        readiness = values.get('readiness_score')
        fatigue = values.get('fatigue')

        if readiness is not None and fatigue is not None:
            # High readiness (>80) with high fatigue (>7) is inconsistent
            if readiness > 80 and fatigue > 7:
                raise ValueError(
                    f"Inconsistent data: High readiness ({readiness}) but high fatigue ({fatigue}). "
                    "Please verify inputs."
                )
            # Low readiness (<40) with low fatigue (<3) is inconsistent
            if readiness < 40 and fatigue < 3:
                raise ValueError(
                    f"Inconsistent data: Low readiness ({readiness}) but low fatigue ({fatigue}). "
                    "Please verify inputs."
                )

        return values

    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "readiness_score": 85,
                "sleep_score": 82,
                "hrv": 55,
                "resting_hr": 58,
                "activity_score": 75,
                "exercise_minutes": 45,
                "fatigue": 4,
                "mood": 7,
                "stress": 3
            }
        }


# ============================================
# Recommendation Schema
# ============================================

class WorkoutRecommendation(BaseModel):
    """Workout recommendation output."""

    workout_type: WorkoutType = Field(
        ...,
        description="Type of workout recommended"
    )
    intensity: Intensity = Field(
        ...,
        description="Recommended intensity level"
    )
    duration_minutes: int = Field(
        ...,
        ge=0,
        le=180,
        description="Recommended workout duration in minutes"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation for the recommendation"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0.0-1.0)"
    )

    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "workout_type": "Upper Body Strength",
                "intensity": "moderate",
                "duration_minutes": 45,
                "explanation": "Your readiness is high and HRV is good. Good day for strength training.",
                "confidence": 0.87
            }
        }


# ============================================
# Training History Schema
# ============================================

class TrainingEntry(BaseModel):
    """Single training session record."""

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the training occurred"
    )
    state: BodyState = Field(
        ...,
        description="Body state before the workout"
    )
    recommendation: Optional[WorkoutRecommendation] = Field(
        default=None,
        description="The recommended workout"
    )
    workout_type: Optional[WorkoutType] = Field(
        default=None,
        description="Actual workout performed (may differ from recommendation)"
    )
    actual_duration: Optional[int] = Field(
        default=None,
        ge=0,
        le=300,
        description="Actual workout duration in minutes"
    )
    source: DataSource = Field(
        default=DataSource.MANUAL_UPLOAD,
        description="Source of this entry"
    )
    user_feedback: Optional[str] = Field(
        default=None,
        description="User's feedback on the workout"
    )
    thumbs_up: Optional[bool] = Field(
        default=None,
        description="User approval (True) or disapproval (False)"
    )

    class Config:
        use_enum_values = True


# ============================================
# User Profile Schema
# ============================================

class UserProfile(BaseModel):
    """User profile information."""

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique user identifier"
    )
    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="User age in years"
    )
    weight: float = Field(
        ...,
        ge=30.0,
        le=300.0,
        description="User weight in kg"
    )
    height: float = Field(
        ...,
        ge=100.0,
        le=250.0,
        description="User height in cm"
    )
    fitness_goal: FitnessGoal = Field(
        ...,
        description="Primary fitness goal"
    )
    training_experience: Optional[Literal["beginner", "intermediate", "advanced"]] = Field(
        default="beginner",
        description="Training experience level"
    )

    @validator('age')
    def validate_age(cls, v):
        """Ensure user is adult."""
        if v < 18:
            raise ValueError("User must be 18 or older to use this system.")
        return v

    @property
    def bmi(self) -> float:
        """Calculate BMI."""
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 2)

    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "user_id": "user_001",
                "age": 30,
                "weight": 75.0,
                "height": 175.0,
                "fitness_goal": "strength",
                "training_experience": "intermediate"
            }
        }


# ============================================
# API Request/Response Schemas
# ============================================

class RecommendationRequest(BaseModel):
    """API request for workout recommendation."""

    user_id: str = Field(..., description="User identifier")
    body_state: BodyState = Field(..., description="Current body state")
    fitness_goal: Optional[FitnessGoal] = Field(
        default=FitnessGoal.GENERAL_FITNESS,
        description="Fitness goal override"
    )

    class Config:
        use_enum_values = True


class RecommendationResponse(BaseModel):
    """API response for workout recommendation."""

    recommendation: WorkoutRecommendation
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

    class Config:
        use_enum_values = True


class FeedbackRequest(BaseModel):
    """API request for user feedback."""

    user_id: str = Field(..., description="User identifier")
    recommendation_id: Optional[str] = None
    thumbs_up: bool = Field(..., description="User approval")
    comment: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional feedback comment"
    )
    actual_workout: Optional[WorkoutType] = None
    actual_duration: Optional[int] = Field(
        default=None,
        ge=0,
        le=300
    )

    class Config:
        use_enum_values = True


class HealthCheckResponse(BaseModel):
    """API health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    services: Dict[str, str] = Field(
        default_factory=lambda: {
            "api": "healthy",
            "database": "healthy",
            "kafka": "healthy"
        }
    )


# ============================================
# Utility Functions
# ============================================

def validate_body_state(data: Dict[str, Any]) -> BodyState:
    """
    Validate body state data and return validated model.

    Args:
        data: Raw body state dictionary

    Returns:
        Validated BodyState instance

    Raises:
        ValidationError: If data is invalid
    """
    return BodyState(**data)


def validate_training_entry(data: Dict[str, Any]) -> TrainingEntry:
    """
    Validate training entry and return validated model.

    Args:
        data: Raw training entry dictionary

    Returns:
        Validated TrainingEntry instance

    Raises:
        ValidationError: If data is invalid
    """
    return TrainingEntry(**data)
