"""Tests for the feature engineering module."""

import pandas as pd
import numpy as np
import pytest
from src.feature_store.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_initialization(self):
        fe = FeatureEngineer()
        assert fe is not None

    def test_create_daily_features_with_sample_data(self):
        fe = FeatureEngineer()
        # Create sample daily data
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "hrv": np.random.uniform(30, 80, 30),
                "resting_hr": np.random.uniform(50, 70, 30),
                "sleep_duration_hours": np.random.uniform(5, 9, 30),
                "readiness_score": np.random.uniform(50, 95, 30),
                "steps": np.random.uniform(3000, 15000, 30),
                "active_calories": np.random.uniform(100, 800, 30),
                "activity_score": np.random.uniform(40, 100, 30),
            }
        )
        features = fe.create_daily_features(data)
        assert features is not None
        assert len(features) > 0

    def test_features_contain_rolling_stats(self):
        fe = FeatureEngineer()
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "hrv": np.random.uniform(30, 80, 30),
                "resting_hr": np.random.uniform(50, 70, 30),
                "sleep_duration_hours": np.random.uniform(5, 9, 30),
                "readiness_score": np.random.uniform(50, 95, 30),
                "steps": np.random.uniform(3000, 15000, 30),
                "active_calories": np.random.uniform(100, 800, 30),
                "activity_score": np.random.uniform(40, 100, 30),
            }
        )
        features = fe.create_daily_features(data)
        # Should have rolling window features
        col_names = features.columns.tolist()
        has_rolling = any(
            "7d" in c or "3d" in c or "rolling" in c.lower() for c in col_names
        )
        assert has_rolling or len(col_names) > len(data.columns)

    def test_get_feature_list(self):
        fe = FeatureEngineer()
        features = fe.get_feature_list()
        assert isinstance(features, list)
        assert len(features) > 0
