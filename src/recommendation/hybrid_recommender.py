"""
Hybrid Recommender

Combines rule-based heuristics with contextual bandits.
"""

import numpy as np
from typing import Dict, List, Optional

from .action_space import ActionSpace, Action
from .contextual_bandits import ContextualBandit
from .reward_fn import RewardFunction
from src.safety.safety_gate import SafetyGate


class HybridRecommender:
    """
    Hybrid recommendation system.
    
    Combines:
    - Rule-based heuristics (baseline)
    - Contextual bandits (RL)
    """
    
    def __init__(self, use_rl: bool = True):
        """
        Initialize hybrid recommender.
        
        Args:
            use_rl: Whether to use RL (bandits) or just rules
        """
        self.action_space = ActionSpace()
        self.reward_fn = RewardFunction()
        self.safety_gate = SafetyGate()
        
        if use_rl:
            self.bandit = ContextualBandit(self.action_space, self.reward_fn)
        else:
            self.bandit = None
    
    def recommend(self, state: Dict, use_rl: Optional[bool] = None) -> Dict:
        """
        Recommend a training plan.
        
        Args:
            state: Daily state dictionary
            use_rl: Override RL usage (None = use default)
        
        Returns:
            Recommendation dictionary
        """
        use_rl = use_rl if use_rl is not None else (self.bandit is not None)
        
        # Safety check
        safety_result = self.safety_gate.check_state(state)
        
        # Get allowed actions
        all_action_ids = list(range(self.action_space.get_action_count()))
        allowed_actions = self.safety_gate.filter_actions(state, all_action_ids)
        
        if not allowed_actions:
            allowed_actions = [0]  # Default to REST
        
        # Select action
        if use_rl and self.bandit:
            # Use contextual bandit
            context = self._state_to_context(state)
            action_id = self.bandit.select_action(context, allowed_actions)
        else:
            # Use rule-based
            action_id = self._rule_based_recommendation(state, allowed_actions)
        
        action = self.action_space.get_action(action_id)
        
        return {
            'action_id': action_id,
            'workout_type': action.workout_type,
            'intensity': action.intensity,
            'duration_minutes': action.duration_minutes,
            'description': action.description,
            'safety_check': {
                'is_safe': safety_result.is_safe,
                'message': safety_result.message,
            },
            'rationale': self._generate_rationale(state, action),
        }
    
    def _state_to_context(self, state: Dict) -> np.ndarray:
        """Convert state to context vector."""
        # Simple context: key features
        features = [
            state.get('readiness_score', 50) / 100.0,
            state.get('sleep_score', 50) / 100.0,
            state.get('activity_score', 50) / 100.0,
            state.get('hrv', 50) / 100.0,
            state.get('resting_hr', 60) / 100.0,
            state.get('fatigue', 5) / 10.0,
            state.get('days_since_training', 1) / 7.0,
        ]
        
        return np.array(features)
    
    def _rule_based_recommendation(self, state: Dict, 
                                  allowed_actions: List[int]) -> int:
        """Rule-based recommendation."""
        readiness = state.get('readiness_score', 50)
        sleep_hours = state.get('sleep_duration_hours', 7)
        fatigue = state.get('fatigue', 5)
        days_since = state.get('days_since_training', 1)
        
        # Rule 1: Very low readiness → REST
        if readiness < 40 or sleep_hours < 5:
            return 0  # REST
        
        # Rule 2: High fatigue → RECOVERY
        if fatigue >= 7:
            recovery_actions = [a for a in allowed_actions 
                              if self.action_space.get_action(a).workout_type == "RECOVERY"]
            if recovery_actions:
                return recovery_actions[0]
        
        # Rule 3: Long rest → Medium intensity
        if days_since >= 3:
            medium_actions = [a for a in allowed_actions
                            if self.action_space.get_action(a).intensity == "MEDIUM"]
            if medium_actions:
                return medium_actions[0]
        
        # Default: Low intensity
        low_actions = [a for a in allowed_actions
                      if self.action_space.get_action(a).intensity == "LOW"]
        if low_actions:
            return low_actions[0]
        
        return allowed_actions[0]
    
    def _generate_rationale(self, state: Dict, action: Action) -> str:
        """Generate explanation for recommendation."""
        readiness = state.get('readiness_score', 50)
        sleep_hours = state.get('sleep_duration_hours', 7)
        
        if action.workout_type == "REST":
            return f"Rest day recommended due to low readiness ({readiness}) or insufficient sleep ({sleep_hours:.1f}h)"
        elif action.intensity == "LOW":
            return f"Low intensity {action.workout_type.lower()} recommended based on current recovery state"
        else:
            return f"{action.intensity} intensity {action.workout_type.lower()} recommended - you're well recovered"
    
    def update(self, action_id: int, state: Dict, reward: float):
        """Update bandit after feedback."""
        if self.bandit:
            self.bandit.update(action_id, reward)

