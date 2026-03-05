"""
Train Contextual Bandits Model

Offline training on historical data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recommendation.hybrid_recommender import HybridRecommender
from src.recommendation.action_space import ActionSpace
from src.feature_store.feature_engineering import FeatureEngineer


def train_model(features_path: str = "data/features/daily_features.parquet",
                output_path: str = "models/bandit_model.pkl"):
    """
    Train contextual bandit model.
    
    Args:
        features_path: Path to feature DataFrame
        output_path: Path to save trained model
    """
    print("=" * 70)
    print("TRAINING CONTEXTUAL BANDIT MODEL")
    print("=" * 70)
    
    # Load features
    if not Path(features_path).exists():
        print(f"✗ Features not found: {features_path}")
        print("Run feature engineering first:")
        print("  python src/feature_store/feature_engineering.py")
        return
    
    features_df = pd.read_parquet(features_path)
    print(f"✓ Loaded features: {len(features_df)} records")
    
    # Initialize recommender
    recommender = HybridRecommender(use_rl=True)
    action_space = ActionSpace()
    
    # Simulate training (in production, use real feedback)
    print("\n" + "=" * 70)
    print("SIMULATED TRAINING")
    print("=" * 70)
    
    # For MVP: simulate based on rules
    # In production, use actual user feedback
    training_episodes = min(100, len(features_df))
    
    for idx in range(training_episodes):
        state = features_df.iloc[idx].to_dict()
        
        # Get recommendation
        recommendation = recommender.recommend(state)
        action_id = recommendation['action_id']
        
        # Simulate feedback (in production, use real feedback)
        # Rule: higher readiness → higher completion probability
        readiness = state.get('readiness_score', 50)
        completion_prob = readiness / 100.0
        
        completion = 1 if np.random.random() < completion_prob else 0
        reward = completion  # Binary reward for MVP
        
        # Update model
        recommender.update(action_id, state, reward)
        
        if (idx + 1) % 20 == 0:
            stats = recommender.bandit.get_statistics()
            print(f"  Episode {idx + 1}/{training_episodes}: "
                  f"Total actions: {sum(stats['action_counts'])}")
    
    # Save model
    import pickle
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(recommender, f)
    
    print(f"\n✓ Model saved to {output_path}")
    
    # Evaluation
    print("\n" + "=" * 70)
    print("MODEL STATISTICS")
    print("=" * 70)
    
    stats = recommender.bandit.get_statistics()
    print(f"\nAction counts:")
    for action_id, count in enumerate(stats['action_counts'][:10]):
        if count > 0:
            action = action_space.get_action(action_id)
            expected = stats['expected_rewards'][action_id]
            print(f"  {action.description}: {count} times, expected reward: {expected:.3f}")


if __name__ == "__main__":
    train_model()

