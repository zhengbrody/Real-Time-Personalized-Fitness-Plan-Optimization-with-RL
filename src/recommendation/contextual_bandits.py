"""
Contextual Bandits Implementation

Implements Thompson Sampling for contextual bandits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats

from .action_space import ActionSpace, Action
from .reward_fn import RewardFunction


class ContextualBandit:
    """
    Contextual Bandit with Thompson Sampling.
    
    Uses Beta-Bernoulli model for binary rewards (completion).
    """
    
    def __init__(self, action_space: ActionSpace, 
                 reward_fn: Optional[RewardFunction] = None):
        """
        Initialize contextual bandit.
        
        Args:
            action_space: Action space
            reward_fn: Reward function (optional)
        """
        self.action_space = action_space
        self.reward_fn = reward_fn or RewardFunction()
        
        # Beta parameters for each action (alpha, beta)
        # Initialize with uniform prior
        self.alpha = np.ones(action_space.get_action_count())
        self.beta = np.ones(action_space.get_action_count())
        
        # Track statistics
        self.action_counts = np.zeros(action_space.get_action_count())
        self.total_rewards = np.zeros(action_space.get_action_count())
    
    def select_action(self, context: np.ndarray, 
                     allowed_actions: Optional[List[int]] = None) -> int:
        """
        Select action using Thompson Sampling.
        
        Args:
            context: Context features (not used in Beta-Bernoulli, but kept for API)
            allowed_actions: List of allowed action IDs (for safety)
        
        Returns:
            Selected action ID
        """
        if allowed_actions is None:
            allowed_actions = list(range(self.action_space.get_action_count()))
        
        # Sample from Beta distribution for each action
        sampled_rewards = []
        for action_id in allowed_actions:
            # Sample from Beta(alpha, beta)
            sample = np.random.beta(self.alpha[action_id], self.beta[action_id])
            sampled_rewards.append((action_id, sample))
        
        # Select action with highest sampled reward
        best_action = max(sampled_rewards, key=lambda x: x[1])[0]
        
        return best_action
    
    def update(self, action_id: int, reward: float):
        """
        Update bandit parameters after observing reward.
        
        Args:
            action_id: Action that was taken
            reward: Observed reward (0-1 for binary, or continuous)
        """
        # For Beta-Bernoulli: treat reward as binary success/failure
        # If reward > 0.5, count as success
        success = 1 if reward > 0.5 else 0
        
        # Update Beta parameters
        self.alpha[action_id] += success
        self.beta[action_id] += (1 - success)
        
        # Update statistics
        self.action_counts[action_id] += 1
        self.total_rewards[action_id] += reward
    
    def get_action_probabilities(self, context: np.ndarray,
                                allowed_actions: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Get action selection probabilities.
        
        Args:
            context: Context features
            allowed_actions: List of allowed action IDs
        
        Returns:
            Dictionary mapping action_id to probability
        """
        if allowed_actions is None:
            allowed_actions = list(range(self.action_space.get_action_count()))
        
        # Expected reward for each action (mean of Beta distribution)
        expected_rewards = {}
        for action_id in allowed_actions:
            expected = self.alpha[action_id] / (self.alpha[action_id] + self.beta[action_id])
            expected_rewards[action_id] = expected
        
        # Normalize to probabilities
        total = sum(expected_rewards.values())
        if total > 0:
            probabilities = {k: v / total for k, v in expected_rewards.items()}
        else:
            probabilities = {k: 1.0 / len(allowed_actions) for k in allowed_actions}
        
        return probabilities
    
    def get_statistics(self) -> Dict:
        """Get bandit statistics."""
        return {
            'action_counts': self.action_counts.tolist(),
            'total_rewards': self.total_rewards.tolist(),
            'expected_rewards': {
                i: self.alpha[i] / (self.alpha[i] + self.beta[i])
                for i in range(len(self.alpha))
            }
        }


class LinearContextualBandit:
    """
    Linear Contextual Bandit with Thompson Sampling.
    
    Uses linear model for continuous rewards.
    More advanced than Beta-Bernoulli.
    """
    
    def __init__(self, action_space: ActionSpace, 
                 feature_dim: int,
                 reward_fn: Optional[RewardFunction] = None):
        """
        Initialize linear contextual bandit.
        
        Args:
            action_space: Action space
            feature_dim: Dimension of context features
            reward_fn: Reward function
        """
        self.action_space = action_space
        self.reward_fn = reward_fn or RewardFunction()
        self.feature_dim = feature_dim
        self.num_actions = action_space.get_action_count()
        
        # Linear model parameters for each action
        # Prior: N(0, I)
        self.theta = np.zeros((self.num_actions, feature_dim))
        self.B = np.array([np.eye(feature_dim) for _ in range(self.num_actions)])
        self.f = np.zeros((self.num_actions, feature_dim))
        
        # Noise variance
        self.sigma = 1.0
        
        # Statistics
        self.action_counts = np.zeros(self.num_actions)
    
    def select_action(self, context: np.ndarray,
                     allowed_actions: Optional[List[int]] = None) -> int:
        """
        Select action using Linear Thompson Sampling.
        
        Args:
            context: Context features (normalized)
            allowed_actions: List of allowed action IDs
        
        Returns:
            Selected action ID
        """
        if allowed_actions is None:
            allowed_actions = list(range(self.num_actions))
        
        # Sample from posterior for each action
        sampled_rewards = []
        for action_id in allowed_actions:
            # Posterior mean
            theta_hat = np.linalg.solve(self.B[action_id], self.f[action_id])
            
            # Sample from posterior
            B_inv = np.linalg.inv(self.B[action_id])
            theta_sample = np.random.multivariate_normal(theta_hat, self.sigma**2 * B_inv)
            
            # Expected reward
            expected_reward = np.dot(theta_sample, context)
            sampled_rewards.append((action_id, expected_reward))
        
        # Select action with highest sampled reward
        best_action = max(sampled_rewards, key=lambda x: x[1])[0]
        
        return best_action
    
    def update(self, action_id: int, context: np.ndarray, reward: float):
        """
        Update bandit parameters.
        
        Args:
            action_id: Action that was taken
            context: Context features
            reward: Observed reward
        """
        # Update B and f (Bayesian linear regression update)
        self.B[action_id] += np.outer(context, context)
        self.f[action_id] += reward * context
        
        # Update statistics
        self.action_counts[action_id] += 1
    
    def get_expected_reward(self, action_id: int, context: np.ndarray) -> float:
        """Get expected reward for an action."""
        theta_hat = np.linalg.solve(self.B[action_id], self.f[action_id])
        return np.dot(theta_hat, context)

