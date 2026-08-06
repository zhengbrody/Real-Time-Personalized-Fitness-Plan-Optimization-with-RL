"""
Tests for the A/B Testing Experiment Framework.

Covers:
  - experiment creation
  - observation recording
  - detecting significant differences (known different distributions)
  - correctly reporting non-significance (same distribution)
  - required sample size calculation
  - confidence interval coverage
  - ExperimentTracker
"""

import math
import pytest
import numpy as np

from src.ab_testing.experiment_framework import ABExperiment, ExperimentTracker

# ── helpers ──────────────────────────────────────────────────────────


def _make_experiment(**kwargs) -> ABExperiment:
    defaults = dict(
        name="test_exp",
        control_strategy="baseline",
        treatment_strategy="variant",
        min_sample_size=30,
        significance_level=0.05,
    )
    defaults.update(kwargs)
    return ABExperiment(**defaults)


# ── test_experiment_creation ─────────────────────────────────────────


class TestExperimentCreation:
    def test_basic_creation(self):
        exp = _make_experiment()
        assert exp.name == "test_exp"
        assert exp.control_strategy == "baseline"
        assert exp.treatment_strategy == "variant"
        assert exp.min_sample_size == 30
        assert exp.significance_level == 0.05

    def test_custom_params(self):
        exp = _make_experiment(
            name="custom",
            min_sample_size=200,
            significance_level=0.01,
        )
        assert exp.min_sample_size == 200
        assert exp.significance_level == 0.01

    def test_invalid_significance_level(self):
        with pytest.raises(ValueError):
            _make_experiment(significance_level=0.0)
        with pytest.raises(ValueError):
            _make_experiment(significance_level=1.0)
        with pytest.raises(ValueError):
            _make_experiment(significance_level=-0.1)

    def test_invalid_min_sample_size(self):
        with pytest.raises(ValueError):
            _make_experiment(min_sample_size=1)


# ── test_add_observations ───────────────────────────────────────────


class TestAddObservations:
    def test_add_control_observation(self):
        exp = _make_experiment()
        exp.add_observation("control", 0.5)
        assert len(exp._control_rewards) == 1
        assert exp._control_rewards[0] == 0.5

    def test_add_treatment_observation(self):
        exp = _make_experiment()
        exp.add_observation("treatment", 0.8, metadata={"ep": 1})
        assert len(exp._treatment_rewards) == 1
        assert exp._treatment_metadata[0] == {"ep": 1}

    def test_invalid_group(self):
        exp = _make_experiment()
        with pytest.raises(ValueError, match="group must be"):
            exp.add_observation("unknown_group", 0.5)

    def test_multiple_observations(self):
        exp = _make_experiment()
        for i in range(50):
            exp.add_observation("control", float(i))
            exp.add_observation("treatment", float(i) * 2)
        assert len(exp._control_rewards) == 50
        assert len(exp._treatment_rewards) == 50

    def test_results_with_insufficient_data(self):
        exp = _make_experiment()
        exp.add_observation("control", 1.0)
        # Only 1 observation per group -- not enough
        results = exp.get_results()
        assert results["significant"] is False
        assert results["p_value"] is None


# ── test_significant_difference ─────────────────────────────────────


class TestSignificantDifference:
    def test_clearly_different_distributions(self):
        """Control ~ N(0, 0.5), Treatment ~ N(1, 0.5).  Should be significant."""
        rng = np.random.default_rng(12345)
        exp = _make_experiment(min_sample_size=30)

        n = 200
        for val in rng.normal(0.0, 0.5, size=n):
            exp.add_observation("control", float(val))
        for val in rng.normal(1.0, 0.5, size=n):
            exp.add_observation("treatment", float(val))

        results = exp.get_results()
        assert results["significant"] is True
        assert results["p_value"] < 0.001
        assert (
            results["lift_pct"] > 0
        )  # treatment mean > 0 > control mean... lift_pct may be unusual
        assert results["effect_size_cohens_d"] > 1.0  # large effect

    def test_is_significant_method(self):
        rng = np.random.default_rng(99)
        exp = _make_experiment(min_sample_size=30)
        for val in rng.normal(0, 1, size=100):
            exp.add_observation("control", float(val))
        for val in rng.normal(2, 1, size=100):
            exp.add_observation("treatment", float(val))
        assert exp.is_significant() is True


# ── test_not_significant ────────────────────────────────────────────


class TestNotSignificant:
    def test_same_distribution(self):
        """Both groups ~ N(0.5, 0.3). Should NOT be significant."""
        rng = np.random.default_rng(42)
        exp = _make_experiment(min_sample_size=30)

        n = 200
        for val in rng.normal(0.5, 0.3, size=n):
            exp.add_observation("control", float(val))
        for val in rng.normal(0.5, 0.3, size=n):
            exp.add_observation("treatment", float(val))

        results = exp.get_results()
        assert results["significant"] is False
        assert results["p_value"] > 0.05
        # Cohen's d should be small
        assert abs(results["effect_size_cohens_d"]) < 0.3

    def test_below_min_sample_size(self):
        """Even if p < 0.05, not significant if n < min_sample_size."""
        rng = np.random.default_rng(7)
        exp = _make_experiment(min_sample_size=500)

        for val in rng.normal(0, 0.1, size=50):
            exp.add_observation("control", float(val))
        for val in rng.normal(5, 0.1, size=50):
            exp.add_observation("treatment", float(val))

        results = exp.get_results()
        assert results["p_value"] < 0.05
        assert results["significant"] is False  # min_sample_size=500 not met


# ── test_required_sample_size ───────────────────────────────────────


class TestRequiredSampleSize:
    def test_basic_calculation(self):
        exp = _make_experiment(min_sample_size=10)
        # Seed with some data so sigma can be estimated
        rng = np.random.default_rng(0)
        for val in rng.normal(0, 1, size=50):
            exp.add_observation("control", float(val))
        for val in rng.normal(0, 1, size=50):
            exp.add_observation("treatment", float(val))

        n = exp.get_required_sample_size(mde=0.5, power=0.8)
        # For d~0.5 with sigma~1, theory says ~64 per group
        assert 30 < n < 200

    def test_smaller_mde_needs_more_samples(self):
        exp = _make_experiment(min_sample_size=10)
        rng = np.random.default_rng(1)
        for val in rng.normal(0, 1, size=50):
            exp.add_observation("control", float(val))

        n_large_mde = exp.get_required_sample_size(mde=0.5, power=0.8)
        n_small_mde = exp.get_required_sample_size(mde=0.1, power=0.8)
        assert n_small_mde > n_large_mde

    def test_higher_power_needs_more_samples(self):
        exp = _make_experiment(min_sample_size=10)
        rng = np.random.default_rng(2)
        for val in rng.normal(0, 1, size=50):
            exp.add_observation("control", float(val))

        n_low = exp.get_required_sample_size(mde=0.3, power=0.6)
        n_high = exp.get_required_sample_size(mde=0.3, power=0.95)
        assert n_high > n_low

    def test_invalid_mde(self):
        exp = _make_experiment()
        with pytest.raises(ValueError):
            exp.get_required_sample_size(mde=0.0)
        with pytest.raises(ValueError):
            exp.get_required_sample_size(mde=-1.0)

    def test_no_data_uses_default_sigma(self):
        """With no observations, sigma defaults to 1.0."""
        exp = _make_experiment(min_sample_size=10)
        n = exp.get_required_sample_size(mde=0.5, power=0.8)
        assert n > 10  # should still return a reasonable number


# ── test_confidence_interval ────────────────────────────────────────


class TestConfidenceInterval:
    def test_contains_true_diff(self):
        """
        Repeat many trials: the 95% CI should contain the true difference
        in roughly 95% of cases (allow some sampling variance).
        """
        true_ctrl_mean = 1.0
        true_treat_mean = 1.5
        true_diff = true_treat_mean - true_ctrl_mean
        n_trials = 200
        n_per_group = 100
        covers = 0

        for trial in range(n_trials):
            rng = np.random.default_rng(trial)
            exp = _make_experiment(min_sample_size=10)

            for val in rng.normal(true_ctrl_mean, 0.5, size=n_per_group):
                exp.add_observation("control", float(val))
            for val in rng.normal(true_treat_mean, 0.5, size=n_per_group):
                exp.add_observation("treatment", float(val))

            results = exp.get_results()
            ci = results["confidence_interval_95"]
            if ci[0] <= true_diff <= ci[1]:
                covers += 1

        coverage = covers / n_trials
        # Should be close to 0.95; allow [0.88, 1.0] for sampling noise
        assert 0.88 <= coverage <= 1.0, f"Coverage was {coverage:.2f}"

    def test_ci_excludes_zero_when_significant(self):
        """If significant, the CI for the difference should not contain 0."""
        rng = np.random.default_rng(55)
        exp = _make_experiment(min_sample_size=30)

        for val in rng.normal(0, 0.5, size=200):
            exp.add_observation("control", float(val))
        for val in rng.normal(1, 0.5, size=200):
            exp.add_observation("treatment", float(val))

        results = exp.get_results()
        assert results["significant"] is True
        ci = results["confidence_interval_95"]
        assert ci[0] > 0 or ci[1] < 0  # zero not in CI


# ── test_experiment_tracker ─────────────────────────────────────────


class TestExperimentTracker:
    def test_create_and_get(self):
        tracker = ExperimentTracker()
        exp = tracker.create_experiment("exp1", "ctrl", "treat")
        assert isinstance(exp, ABExperiment)
        assert tracker.get_experiment("exp1") is exp

    def test_duplicate_name_raises(self):
        tracker = ExperimentTracker()
        tracker.create_experiment("exp1", "ctrl", "treat")
        with pytest.raises(ValueError, match="already exists"):
            tracker.create_experiment("exp1", "ctrl", "treat2")

    def test_get_nonexistent_raises(self):
        tracker = ExperimentTracker()
        with pytest.raises(KeyError):
            tracker.get_experiment("nope")

    def test_list_experiments(self):
        tracker = ExperimentTracker()
        tracker.create_experiment("a", "c1", "t1")
        tracker.create_experiment("b", "c2", "t2")
        names = tracker.list_experiments()
        assert set(names) == {"a", "b"}

    def test_get_summary(self):
        tracker = ExperimentTracker()
        exp = tracker.create_experiment("exp1", "ctrl", "treat", min_sample_size=2)

        rng = np.random.default_rng(0)
        for val in rng.normal(0, 1, size=10):
            exp.add_observation("control", float(val))
        for val in rng.normal(1, 1, size=10):
            exp.add_observation("treatment", float(val))

        summary = tracker.get_summary()
        assert "exp1" in summary
        assert "p_value" in summary["exp1"]
        assert summary["exp1"]["sample_sizes"]["control"] == 10

    def test_kwargs_forwarded(self):
        tracker = ExperimentTracker()
        exp = tracker.create_experiment(
            "strict",
            "c",
            "t",
            min_sample_size=500,
            significance_level=0.01,
        )
        assert exp.min_sample_size == 500
        assert exp.significance_level == 0.01
