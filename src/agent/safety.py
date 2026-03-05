"""
Safety Guardrails for AI Coach Agent

This module implements safety checks to prevent dangerous recommendations
and ensure user safety.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .state import DailyState


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    is_safe: bool
    risk_level: str  # "low", "medium", "high", "critical"
    message: str
    recommended_action: str


class SafetyGuardrails:
    """
    Safety guardrails for the AI Coach Agent.
    
    Implements hard rules and simple models to prevent dangerous recommendations.
    """
    
    # Thresholds
    MIN_HRV = 20  # ms
    MAX_RESTING_HR = 100  # bpm
    MIN_SLEEP_HOURS = 4  # hours
    MAX_CONSECUTIVE_HIGH_INTENSITY = 3  # days
    MAX_FATIGUE_FOR_TRAINING = 8  # out of 10
    
    def __init__(self):
        """Initialize safety guardrails."""
        pass
    
    def check_state(self, state: DailyState) -> SafetyCheckResult:
        """
        Perform comprehensive safety check on daily state.
        
        Args:
            state: DailyState to check
        
        Returns:
            SafetyCheckResult
        """
        checks = [
            self._check_physiological_signals(state),
            self._check_overtraining_risk(state),
            self._check_fatigue_level(state),
            self._check_sleep_recovery(state),
            self._check_injury_risks(state),
        ]
        
        # Find highest risk level
        critical_checks = [c for c in checks if c.risk_level == "critical"]
        high_checks = [c for c in checks if c.risk_level == "high"]
        medium_checks = [c for c in checks if c.risk_level == "medium"]
        
        if critical_checks:
            return critical_checks[0]
        elif high_checks:
            return high_checks[0]
        elif medium_checks:
            return medium_checks[0]
        else:
            return SafetyCheckResult(
                is_safe=True,
                risk_level="low",
                message="All safety checks passed",
                recommended_action="proceed_with_plan"
            )
    
    def _check_physiological_signals(self, state: DailyState) -> SafetyCheckResult:
        """Check for abnormal physiological signals."""
        issues = []
        
        if state.resting_heart_rate and state.resting_heart_rate > self.MAX_RESTING_HR:
            issues.append(f"Resting heart rate elevated ({state.resting_heart_rate} bpm)")
        
        if state.hrv and state.hrv < self.MIN_HRV:
            issues.append(f"HRV very low ({state.hrv} ms)")
        
        if issues:
            return SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                message="Abnormal physiological signals detected: " + "; ".join(issues),
                recommended_action="rest_day_or_light_activity"
            )
        
        return SafetyCheckResult(
            is_safe=True,
            risk_level="low",
            message="Physiological signals normal",
            recommended_action="proceed"
        )
    
    def _check_overtraining_risk(self, state: DailyState) -> SafetyCheckResult:
        """Check for overtraining risk."""
        if state.overtraining_risk:
            return SafetyCheckResult(
                is_safe=False,
                risk_level="critical",
                message="Overtraining risk detected. Multiple indicators suggest excessive training load.",
                recommended_action="mandatory_rest_day"
            )
        
        # Check consecutive high-intensity days
        if (state.training_frequency_last_week and 
            state.training_frequency_last_week >= self.MAX_CONSECUTIVE_HIGH_INTENSITY):
            return SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                message=f"High training frequency detected ({state.training_frequency_last_week} days/week)",
                recommended_action="reduce_intensity_or_rest"
            )
        
        return SafetyCheckResult(
            is_safe=True,
            risk_level="low",
            message="No overtraining risk detected",
            recommended_action="proceed"
        )
    
    def _check_fatigue_level(self, state: DailyState) -> SafetyCheckResult:
        """Check fatigue level."""
        if state.fatigue_level and state.fatigue_level >= self.MAX_FATIGUE_FOR_TRAINING:
            return SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                message=f"High fatigue level ({state.fatigue_level}/10). Training may be counterproductive.",
                recommended_action="rest_day_or_recovery_session"
            )
        
        return SafetyCheckResult(
            is_safe=True,
            risk_level="low",
            message="Fatigue level acceptable",
            recommended_action="proceed"
        )
    
    def _check_sleep_recovery(self, state: DailyState) -> SafetyCheckResult:
        """Check sleep and recovery."""
        if state.sleep_duration_hours and state.sleep_duration_hours < self.MIN_SLEEP_HOURS:
            return SafetyCheckResult(
                is_safe=False,
                risk_level="medium",
                message=f"Insufficient sleep ({state.sleep_duration_hours} hours). Recovery may be compromised.",
                recommended_action="light_training_or_rest"
            )
        
        if state.readiness_score and state.readiness_score < 50:
            return SafetyCheckResult(
                is_safe=False,
                risk_level="medium",
                message=f"Low readiness score ({state.readiness_score}/100). Consider lighter training.",
                recommended_action="reduce_intensity"
            )
        
        return SafetyCheckResult(
            is_safe=True,
            risk_level="low",
            message="Sleep and recovery adequate",
            recommended_action="proceed"
        )
    
    def _check_injury_risks(self, state: DailyState) -> SafetyCheckResult:
        """Check for injury-related risks."""
        if state.muscle_soreness and state.muscle_soreness >= 8:
            return SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                message=f"High muscle soreness ({state.muscle_soreness}/10). Risk of injury if training intensively.",
                recommended_action="active_recovery_or_rest"
            )
        
        return SafetyCheckResult(
            is_safe=True,
            risk_level="low",
            message="No injury risks detected",
            recommended_action="proceed"
        )
    
    def check_plan_safety(self, plan: Dict, state: DailyState) -> SafetyCheckResult:
        """
        Check if a specific training plan is safe given current state.
        
        Args:
            plan: Training plan dictionary
            state: Current daily state
        
        Returns:
            SafetyCheckResult
        """
        # Check plan intensity against state
        plan_intensity = plan.get('intensity', 'moderate')
        plan_volume = plan.get('volume', 'moderate')
        
        # If high intensity and state shows fatigue, flag it
        if plan_intensity == 'high' and state.fatigue_level and state.fatigue_level >= 7:
            return SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                message="High intensity plan requested but fatigue level is elevated",
                recommended_action="reduce_intensity"
            )
        
        # Check for injury-related exercises
        if state.injury_history:
            plan_exercises = plan.get('exercises', [])
            for exercise in plan_exercises:
                exercise_name = exercise.get('name', '').lower()
                for injury in state.injury_history:
                    if injury.lower() in exercise_name:
                        return SafetyCheckResult(
                            is_safe=False,
                            risk_level="critical",
                            message=f"Plan includes exercise that may aggravate injury history: {exercise_name}",
                            recommended_action="modify_plan"
                        )
        
        return SafetyCheckResult(
            is_safe=True,
            risk_level="low",
            message="Plan is safe given current state",
            recommended_action="proceed"
        )
    
    def escalate_safety_alert(self, state: DailyState, reason: str) -> Dict:
        """
        Escalate a safety alert - for critical situations.
        
        Args:
            state: Current state
            reason: Reason for escalation
        
        Returns:
            Alert dictionary
        """
        return {
            'type': 'safety_alert',
            'severity': 'critical',
            'user_id': state.user_id,
            'timestamp': state.date,
            'reason': reason,
            'recommended_action': 'stop_training_and_consult_professional',
            'message': (
                "⚠️ SAFETY ALERT: " + reason + 
                "\n\nPlease stop training and consult with a healthcare professional if you experience: "
                "chest pain, dizziness, severe discomfort, or abnormal heart rate patterns."
            ),
        }

