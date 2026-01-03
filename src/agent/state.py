"""
Daily State Management for AI Coach Agent

This module handles the construction of daily state from feature store
and real-time data streams.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd


@dataclass
class DailyState:
    """
    Daily state representation for the AI Coach Agent.
    
    Contains aggregated features from wearable devices, training history,
    and user feedback - safe for LLM consumption.
    """
    user_id: str
    date: str
    
    # Physiological metrics
    hrv: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    sleep_duration_hours: Optional[float] = None
    sleep_quality_score: Optional[float] = None
    readiness_score: Optional[float] = None
    
    # Activity metrics
    steps: Optional[int] = None
    active_calories: Optional[float] = None
    
    # Training history
    days_since_last_training: Optional[int] = None
    training_frequency_last_week: Optional[int] = None
    avg_rpe_last_week: Optional[float] = None
    completion_rate_last_week: Optional[float] = None
    
    # User feedback
    fatigue_level: Optional[int] = None  # 1-10
    motivation_level: Optional[int] = None  # 1-10
    muscle_soreness: Optional[int] = None  # 1-10
    stress_level: Optional[int] = None  # 1-5
    mood_score: Optional[int] = None  # 1-5
    
    # Goals
    primary_goal: Optional[str] = None  # "muscle_gain", "fat_loss", "endurance", "strength"
    current_phase: Optional[str] = None  # "bulking", "cutting", "maintenance"
    
    # Safety flags
    injury_history: List[str] = None
    overtraining_risk: bool = False
    
    def __post_init__(self):
        if self.injury_history is None:
            self.injury_history = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for LLM consumption."""
        return {
            'user_id': self.user_id,
            'date': self.date,
            'physiological': {
                'hrv': self.hrv,
                'resting_heart_rate': self.resting_heart_rate,
                'sleep_duration_hours': self.sleep_duration_hours,
                'sleep_quality_score': self.sleep_quality_score,
                'readiness_score': self.readiness_score,
            },
            'activity': {
                'steps': self.steps,
                'active_calories': self.active_calories,
            },
            'training': {
                'days_since_last_training': self.days_since_last_training,
                'training_frequency_last_week': self.training_frequency_last_week,
                'avg_rpe_last_week': self.avg_rpe_last_week,
                'completion_rate_last_week': self.completion_rate_last_week,
            },
            'user_feedback': {
                'fatigue_level': self.fatigue_level,
                'motivation_level': self.motivation_level,
                'muscle_soreness': self.muscle_soreness,
                'stress_level': self.stress_level,
                'mood_score': self.mood_score,
            },
            'goals': {
                'primary_goal': self.primary_goal,
                'current_phase': self.current_phase,
            },
            'safety': {
                'injury_history': self.injury_history,
                'overtraining_risk': self.overtraining_risk,
            },
        }
    
    def to_natural_language(self) -> str:
        """Convert to natural language summary for LLM."""
        parts = []
        
        if self.readiness_score:
            parts.append(f"Readiness score: {self.readiness_score}/100")
        if self.sleep_quality_score:
            parts.append(f"Sleep quality: {self.sleep_quality_score}/100")
        if self.resting_heart_rate:
            parts.append(f"Resting heart rate: {self.resting_heart_rate} bpm")
        if self.days_since_last_training is not None:
            parts.append(f"Last trained {self.days_since_last_training} days ago")
        if self.fatigue_level:
            parts.append(f"Fatigue level: {self.fatigue_level}/10")
        if self.motivation_level:
            parts.append(f"Motivation level: {self.motivation_level}/10")
        if self.mood_score:
            parts.append(f"Mood score: {self.mood_score}/5")
        
        return ". ".join(parts) if parts else "No data available"


class DailyStateBuilder:
    """Builds DailyState from feature store and data sources."""
    
    def __init__(self, feature_store_client=None):
        """
        Initialize state builder.
        
        Args:
            feature_store_client: Feast feature store client (optional)
        """
        self.feature_store = feature_store_client
    
    def build_state(self, user_id: str, date: Optional[str] = None) -> DailyState:
        """
        Build daily state for a user.
        
        Args:
            user_id: User identifier
            date: Date in YYYY-MM-DD format (defaults to today)
        
        Returns:
            DailyState object
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # In production, fetch from feature store
        # For now, return a template state
        state = DailyState(
            user_id=user_id,
            date=date,
            # These would be fetched from feature store in production
            hrv=None,
            resting_heart_rate=None,
            sleep_duration_hours=None,
            sleep_quality_score=None,
            readiness_score=None,
            steps=None,
            active_calories=None,
            days_since_last_training=None,
            training_frequency_last_week=None,
            avg_rpe_last_week=None,
            completion_rate_last_week=None,
            fatigue_level=None,
            motivation_level=None,
            muscle_soreness=None,
            stress_level=None,
            mood_score=None,
            primary_goal=None,
            current_phase=None,
            injury_history=[],
            overtraining_risk=False,
        )
        
        return state
    
    def update_from_feature_store(self, state: DailyState) -> DailyState:
        """
        Update state from feature store.
        
        Args:
            state: DailyState to update
        
        Returns:
            Updated DailyState
        """
        # TODO: Implement feature store fetching
        # Example:
        # features = self.feature_store.get_online_features(...)
        # state.hrv = features.get('hrv')
        # state.resting_heart_rate = features.get('resting_heart_rate')
        # etc.
        
        return state

