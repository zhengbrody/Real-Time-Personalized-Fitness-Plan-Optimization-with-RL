"""
Feature Engineering for RL Model

Creates features for contextual bandits from daily unified data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class FeatureEngineer:
    """Engineer features for RL model."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def create_daily_features(self, unified_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily features from unified data.
        
        Args:
            unified_df: Unified daily DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        df = unified_df.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Recovery features
        df = self._create_recovery_features(df)
        
        # Load features
        df = self._create_load_features(df)
        
        # Consistency features
        df = self._create_consistency_features(df)
        
        # Temporal features
        df = self._create_temporal_features(df)
        
        # Missing indicators
        df = self._create_missing_indicators(df)
        
        return df
    
    def _create_recovery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create recovery-related features."""
        # HRV features
        if 'hrv' in df.columns:
            df['hrv_7d_mean'] = df['hrv'].rolling(window=7, min_periods=1).mean()
            df['hrv_7d_zscore'] = (df['hrv'] - df['hrv_7d_mean']) / (df['hrv'].rolling(window=7, min_periods=1).std() + 1e-6)
            df['hrv_trend'] = df['hrv'].rolling(window=3, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        
        # Resting HR features
        if 'resting_hr' in df.columns:
            baseline_hr = df['resting_hr'].quantile(0.5)  # Median as baseline
            df['rhr_deviation'] = df['resting_hr'] - baseline_hr
            df['rhr_7d_mean'] = df['resting_hr'].rolling(window=7, min_periods=1).mean()
        
        # Sleep features
        if 'sleep_duration_hours' in df.columns:
            df['sleep_debt'] = df['sleep_duration_hours'].rolling(window=3, min_periods=1).apply(
                lambda x: max(0, 8 - x.mean()) if len(x) > 0 else 0
            )
            df['sleep_7d_mean'] = df['sleep_duration_hours'].rolling(window=7, min_periods=1).mean()
        
        # Readiness score
        if 'readiness_score' in df.columns:
            df['readiness_7d_mean'] = df['readiness_score'].rolling(window=7, min_periods=1).mean()
            df['readiness_trend'] = df['readiness_score'].rolling(window=3, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        return df
    
    def _create_load_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create training load features."""
        # Steps features
        if 'steps' in df.columns:
            df['steps_7d_sum'] = df['steps'].rolling(window=7, min_periods=1).sum()
            df['steps_7d_mean'] = df['steps'].rolling(window=7, min_periods=1).mean()
        
        # Calories features
        if 'active_calories' in df.columns:
            df['calories_7d_sum'] = df['active_calories'].rolling(window=7, min_periods=1).sum()
            df['calories_3d_sum'] = df['active_calories'].rolling(window=3, min_periods=1).sum()
        
        # Activity score
        if 'activity_score' in df.columns:
            df['activity_7d_mean'] = df['activity_score'].rolling(window=7, min_periods=1).mean()
        
        # Acute:Chronic workload ratio (simplified)
        if 'active_calories' in df.columns:
            acute = df['active_calories'].rolling(window=3, min_periods=1).mean()
            chronic = df['active_calories'].rolling(window=7, min_periods=1).mean()
            df['acwr'] = acute / (chronic + 1e-6)
        
        return df
    
    def _create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create consistency/training history features."""
        # Completion rate (placeholder - will be filled from training logs)
        if 'completion' not in df.columns:
            df['completion'] = np.nan
        
        if 'completion' in df.columns:
            df['completion_7d_rate'] = df['completion'].rolling(window=7, min_periods=1).mean()
        
        # Streak (consecutive training days)
        if 'completion' in df.columns:
            df['streak'] = (df['completion'] == 1).groupby((df['completion'] != 1).cumsum()).cumsum()
        
        # Days since last training
        if 'completion' in df.columns:
            df['days_since_training'] = df.groupby((df['completion'] == 1).cumsum()).cumcount()
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal context features."""
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        
        return df
    
    def _create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create missing value indicators."""
        key_features = ['hrv', 'resting_hr', 'sleep_duration_hours', 'readiness_score', 'steps']
        
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_missing'] = df[feat].isna().astype(int)
        
        return df
    
    def get_feature_list(self) -> List[str]:
        """Get list of all engineered features."""
        return [
            # Recovery
            'hrv', 'hrv_7d_mean', 'hrv_7d_zscore', 'hrv_trend',
            'resting_hr', 'rhr_deviation', 'rhr_7d_mean',
            'sleep_duration_hours', 'sleep_debt', 'sleep_7d_mean',
            'readiness_score', 'readiness_7d_mean', 'readiness_trend',
            # Load
            'steps', 'steps_7d_sum', 'steps_7d_mean',
            'active_calories', 'calories_7d_sum', 'calories_3d_sum',
            'activity_score', 'activity_7d_mean', 'acwr',
            # Consistency
            'completion_7d_rate', 'streak', 'days_since_training',
            # Temporal
            'day_of_week', 'is_weekend', 'day_of_month', 'month',
            # Missing indicators
            'hrv_missing', 'resting_hr_missing', 'sleep_duration_hours_missing',
        ]


def main():
    """Main feature engineering function."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    # Load unified data
    unified_path = Path("data/processed/unified_daily.parquet")
    
    if not unified_path.exists():
        print("✗ Unified data not found. Run preprocessing first:")
        print("  python src/data_collection/preprocess.py")
        return
    
    unified_df = pd.read_parquet(unified_path)
    print(f"✓ Loaded unified data: {len(unified_df)} records")
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.create_daily_features(unified_df)
    
    print(f"\n✓ Created features: {len(features_df.columns)} columns")
    print(f"  Feature list: {len(engineer.get_feature_list())} features")
    
    # Save
    output_path = Path("data/features/daily_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved features to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    print(f"\nTotal features: {len(features_df.columns)}")
    print(f"\nSample features:")
    for feat in engineer.get_feature_list()[:15]:
        if feat in features_df.columns:
            non_null = features_df[feat].notna().sum()
            print(f"  ✓ {feat}: {non_null}/{len(features_df)} non-null")


if __name__ == "__main__":
    main()

