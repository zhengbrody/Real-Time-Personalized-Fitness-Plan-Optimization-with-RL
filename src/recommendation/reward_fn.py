"""
Reward Function Definition

Defines what "good" means for the RL system.
"""

import numpy as np
from typing import Dict, Optional


class RewardFunction:
    """Reward function for RL training."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize reward function.
        
        Args:
            weights: Weights for different reward components
        """
        self.weights = weights or {
            'completion': 1.0,
            'adherence': 0.5,
            'recovery_change': -1.0,  # Negative: recovery decline is bad
            'satisfaction': 0.3,
            'overtraining_penalty': -2.0,
        }
    
    def compute_reward(self, 
                      completion: float,
                      adherence_ratio: float = 1.0,
                      recovery_change: float = 0.0,
                      satisfaction: float = 0.5,
                      overtraining: bool = False) -> float:
        """
        Compute reward from outcomes.
        
        Args:
            completion: 1.0 if completed, 0.0 otherwise
            adherence_ratio: Ratio of actual to planned (0-1)
            recovery_change: Change in recovery (positive = better)
            satisfaction: User satisfaction (0-1)
            overtraining: Whether overtraining occurred
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Completion reward
        reward += self.weights['completion'] * completion
        
        # Adherence reward
        reward += self.weights['adherence'] * adherence_ratio
        
        # Recovery change (negative weight: decline is bad)
        reward += self.weights['recovery_change'] * recovery_change
        
        # Satisfaction
        reward += self.weights['satisfaction'] * satisfaction
        
        # Overtraining penalty
        if overtraining:
            reward += self.weights['overtraining_penalty']
        
        return reward
    
    def compute_reward_from_dict(self, outcomes: Dict) -> float:
        """
        Compute reward from outcomes dictionary.
        
        Args:
            outcomes: Dictionary with reward components
        
        Returns:
            Reward value
        """
        return self.compute_reward(
            completion=outcomes.get('completion', 0.0),
            adherence_ratio=outcomes.get('adherence_ratio', 1.0),
            recovery_change=outcomes.get('recovery_change', 0.0),
            satisfaction=outcomes.get('satisfaction', 0.5),
            overtraining=outcomes.get('overtraining', False)
        )

