"""Tests for the feature engineering module."""

import pandas as pd
import numpy as np
import pytest
from src.feature_store.feature_engineering import FeatureEngineer, main


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_initialization(self):
        fe = FeatureEngineer()
        assert fe is not None

    def test_create_daily_features_with_sample_data(self):
        fe = FeatureEngineer()
        rng = np.random.default_rng(42)
        # Create sample daily data
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "hrv": rng.uniform(30, 80, 30),
                "resting_hr": rng.uniform(50, 70, 30),
                "sleep_duration_hours": rng.uniform(5, 9, 30),
                "readiness_score": rng.uniform(50, 95, 30),
                "steps": rng.uniform(3000, 15000, 30),
                "active_calories": rng.uniform(100, 800, 30),
                "activity_score": rng.uniform(40, 100, 30),
            }
        )
        features = fe.create_daily_features(data)
        assert features is not None
        assert len(features) > 0

    def test_features_contain_rolling_stats(self):
        fe = FeatureEngineer()
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "hrv": rng.uniform(30, 80, 30),
                "resting_hr": rng.uniform(50, 70, 30),
                "sleep_duration_hours": rng.uniform(5, 9, 30),
                "readiness_score": rng.uniform(50, 95, 30),
                "steps": rng.uniform(3000, 15000, 30),
                "active_calories": rng.uniform(100, 800, 30),
                "activity_score": rng.uniform(40, 100, 30),
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


class TestMain:
    """Tests for the main() function (lines 227-269, 273)."""

    def _make_sample_df(self):
        """Create a sample unified DataFrame for testing."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "hrv": rng.uniform(30, 80, 10),
                "resting_hr": rng.uniform(50, 70, 10),
                "sleep_duration_hours": rng.uniform(5, 9, 10),
                "readiness_score": rng.uniform(50, 95, 10),
                "steps": rng.uniform(3000, 15000, 10),
                "active_calories": rng.uniform(100, 800, 10),
                "activity_score": rng.uniform(40, 100, 10),
            }
        )

    def test_main_data_not_found(self, caplog, tmp_path, monkeypatch):
        """Lines 239-241: When unified data file does not exist, log error and return."""
        import logging

        # chdir to a temp directory where the data file does not exist
        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.INFO):
            main()

        log_text = caplog.text
        assert "FEATURE ENGINEERING" in log_text
        assert "not found" in log_text

    def test_main_success(self, caplog, tmp_path, monkeypatch):
        """Lines 244-269: Successful run loads data, engineers features, saves output."""
        import logging

        sample_df = self._make_sample_df()

        # Create the data file in the expected relative location
        unified_dir = tmp_path / "data" / "processed"
        unified_dir.mkdir(parents=True, exist_ok=True)
        unified_file = unified_dir / "unified_daily.parquet"
        sample_df.to_parquet(unified_file, index=False)

        # chdir to tmp_path so Path("data/processed/unified_daily.parquet") resolves
        monkeypatch.chdir(tmp_path)

        with caplog.at_level(logging.INFO):
            main()

        log_text = caplog.text
        assert "FEATURE ENGINEERING" in log_text
        assert "Loaded unified data" in log_text
        assert "Created features" in log_text
        assert "Saved features" in log_text
        assert "FEATURE SUMMARY" in log_text
        assert "Total features" in log_text
        assert "Sample features" in log_text

        # Verify the output file was actually written
        output_file = tmp_path / "data" / "features" / "daily_features.parquet"
        assert output_file.exists()
        result_df = pd.read_parquet(output_file)
        assert len(result_df) == 10
