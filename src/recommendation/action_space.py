"""
Action Space Definition

Defines what training plans the system can recommend.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Action:
    """Represents a training plan action."""
    action_id: int
    workout_type: str  # REST, RECOVERY, STRENGTH, CARDIO
    intensity: str     # LOW, MEDIUM, HIGH
    duration_minutes: int  # 20, 30, 45
    description: str


class ActionSpace:
    """Defines the action space for the RL system."""
    
    def __init__(self):
        """Initialize action space."""
        self.actions = self._create_action_space()
        self.action_to_id = {self._action_key(a): a.action_id for a in self.actions}
        self.id_to_action = {a.action_id: a for a in self.actions}
    
    def _create_action_space(self) -> List[Action]:
        """Create the action space."""
        actions = []
        action_id = 0
        
        # REST
        actions.append(Action(
            action_id=action_id,
            workout_type="REST",
            intensity="NONE",
            duration_minutes=0,
            description="Rest day - no training"
        ))
        action_id += 1
        
        # RECOVERY (low intensity)
        for duration in [20, 30]:
            actions.append(Action(
                action_id=action_id,
                workout_type="RECOVERY",
                intensity="LOW",
                duration_minutes=duration,
                description=f"Recovery session - {duration} minutes"
            ))
            action_id += 1
        
        # STRENGTH
        for intensity in ["LOW", "MEDIUM", "HIGH"]:
            for duration in [30, 45]:
                actions.append(Action(
                    action_id=action_id,
                    workout_type="STRENGTH",
                    intensity=intensity,
                    duration_minutes=duration,
                    description=f"Strength training - {intensity} intensity, {duration} min"
                ))
                action_id += 1
        
        # CARDIO
        for intensity in ["LOW", "MEDIUM", "HIGH"]:
            for duration in [20, 30, 45]:
                actions.append(Action(
                    action_id=action_id,
                    workout_type="CARDIO",
                    intensity=intensity,
                    duration_minutes=duration,
                    description=f"Cardio - {intensity} intensity, {duration} min"
                ))
                action_id += 1
        
        return actions
    
    def _action_key(self, action: Action) -> str:
        """Create unique key for action."""
        return f"{action.workout_type}_{action.intensity}_{action.duration_minutes}"
    
    def get_action(self, action_id: int) -> Action:
        """Get action by ID."""
        return self.id_to_action[action_id]
    
    def get_action_id(self, workout_type: str, intensity: str, duration: int) -> int:
        """Get action ID from components."""
        key = f"{workout_type}_{intensity}_{duration}"
        return self.action_to_id.get(key, 0)  # Default to REST
    
    def get_all_actions(self) -> List[Action]:
        """Get all actions."""
        return self.actions
    
    def get_action_count(self) -> int:
        """Get total number of actions."""
        return len(self.actions)
    
    def filter_by_safety(self, allowed_types: List[str], 
                        max_intensity: str = "HIGH") -> List[int]:
        """
        Filter actions by safety constraints.
        
        Args:
            allowed_types: List of allowed workout types
            max_intensity: Maximum allowed intensity
        
        Returns:
            List of allowed action IDs
        """
        intensity_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        max_intensity_level = intensity_order.get(max_intensity, 3)
        
        allowed = []
        for action in self.actions:
            if action.workout_type in allowed_types:
                if intensity_order.get(action.intensity, 0) <= max_intensity_level:
                    allowed.append(action.action_id)
        
        return allowed

