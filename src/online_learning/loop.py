"""
Online Learning Loop

Closed-loop: state → action → feedback → update
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from src.recommendation.hybrid_recommender import HybridRecommender
from src.recommendation.reward_fn import RewardFunction


class OnlineLearningLoop:
    """
    Online learning loop for continuous model updates.
    """
    
    def __init__(self, recommender: Optional[HybridRecommender] = None,
                 kafka_producer=None):
        """
        Initialize online learning loop.
        
        Args:
            recommender: Hybrid recommender instance
            kafka_producer: Kafka producer for event logging
        """
        self.recommender = recommender or HybridRecommender()
        self.reward_fn = RewardFunction()
        self.kafka_producer = kafka_producer
        
        # Event log
        self.event_log = []
    
    def process_daily_cycle(self, user_id: str, state: Dict) -> Dict:
        """
        Process one daily cycle: state → action → recommendation.
        
        Args:
            user_id: User identifier
            state: Daily state
        
        Returns:
            Recommendation dictionary
        """
        # Get recommendation
        recommendation = self.recommender.recommend(state)
        
        # Log event
        event = {
            'event_type': 'plan_served',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action_id': recommendation['action_id'],
            'recommendation': recommendation,
        }
        
        self._log_event(event)
        
        return recommendation
    
    def process_feedback(self, user_id: str, action_id: int, 
                        feedback: Dict) -> float:
        """
        Process user feedback and update model.
        
        Args:
            user_id: User identifier
            action_id: Action that was taken
            feedback: Feedback dictionary (completion, RPE, mood, etc.)
        
        Returns:
            Computed reward
        """
        # Compute reward
        reward = self.reward_fn.compute_reward_from_dict(feedback)
        
        # Update recommender
        self.recommender.update(action_id, {}, reward)
        
        # Log event
        event = {
            'event_type': 'feedback_received',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'action_id': action_id,
            'feedback': feedback,
            'reward': reward,
        }
        
        self._log_event(event)
        
        return reward
    
    def _log_event(self, event: Dict):
        """Log event to Kafka and local log."""
        self.event_log.append(event)
        
        # Send to Kafka if available
        if self.kafka_producer:
            try:
                import json
                topic = 'training.user.feedback' if 'feedback' in event['event_type'] else 'training.plan.served'
                self.kafka_producer.send(topic, value=json.dumps(event))
            except Exception as e:
                print(f"Error logging to Kafka: {e}")
    
    def get_event_log(self) -> pd.DataFrame:
        """Get event log as DataFrame."""
        return pd.DataFrame(self.event_log)

