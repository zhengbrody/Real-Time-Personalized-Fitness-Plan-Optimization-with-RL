"""
Tests for convergence analysis detection methods.

Uses synthetic reward data with known convergence points to validate
each detection method.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.convergence_analysis import (
    detect_convergence_cusum,
    detect_convergence_plateau,
    detect_convergence_rolling_mean,
    rolling_mean,
    rolling_std,
)

# ───────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────


def _make_step_rewards(
    n: int = 1000,
    change_point: int = 200,
    low_mean: float = 0.3,
    high_mean: float = 0.7,
    noise_std: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic rewards with a clear step change.

    Before *change_point*: rewards ~ N(low_mean, noise_std)
    After  *change_point*: rewards ~ N(high_mean, noise_std)
    """
    rng = np.random.default_rng(seed)
    rewards = np.empty(n)
    rewards[:change_point] = rng.normal(low_mean, noise_std, change_point)
    rewards[change_point:] = rng.normal(high_mean, noise_std, n - change_point)
    return rewards


def _make_constant_rewards(
    n: int = 1000, mean: float = 0.6, noise_std: float = 0.15, seed: int = 42
) -> np.ndarray:
    """Already-converged: constant mean throughout."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, noise_std, n)


def _make_never_converges(n: int = 1000, seed: int = 42) -> np.ndarray:
    """Monotonically increasing mean -- never truly converges."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0.0, 1.0, n)
    noise = rng.normal(0, 0.1, n)
    return trend + noise


def _make_gradual_convergence(
    n: int = 1000,
    convergence_ep: int = 150,
    asymptote: float = 0.7,
    noise_std: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """
    Exponential approach to asymptote.

    mean(t) = asymptote * (1 - exp(-t / tau))
    where tau is chosen so the signal is ~95% of asymptote at convergence_ep.
    """
    rng = np.random.default_rng(seed)
    tau = convergence_ep / 3.0  # 95% at convergence_ep
    t = np.arange(n, dtype=float)
    signal = asymptote * (1.0 - np.exp(-t / tau))
    return signal + rng.normal(0, noise_std, n)


# ───────────────────────────────────────────────────────
# Tests for rolling_mean / rolling_std helpers
# ───────────────────────────────────────────────────────


class TestRollingHelpers:
    def test_rolling_mean_constant(self):
        values = np.ones(100) * 5.0
        rm = rolling_mean(values, window=10)
        np.testing.assert_allclose(rm, 5.0)

    def test_rolling_mean_window_1(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rm = rolling_mean(values, window=1)
        np.testing.assert_allclose(rm, values)

    def test_rolling_mean_length(self):
        values = np.random.default_rng(0).normal(0, 1, 500)
        rm = rolling_mean(values, window=50)
        assert len(rm) == len(values)

    def test_rolling_std_constant(self):
        values = np.ones(100) * 3.0
        rstd = rolling_std(values, window=10)
        np.testing.assert_allclose(rstd, 0.0, atol=1e-10)

    def test_rolling_std_positive(self):
        values = np.random.default_rng(0).normal(0, 1, 200)
        rstd = rolling_std(values, window=50)
        # After the window fills, std should be positive
        assert all(rstd[50:] > 0)


# ───────────────────────────────────────────────────────
# Tests for Method 1: Rolling mean threshold
# ───────────────────────────────────────────────────────


class TestRollingMeanThreshold:
    def test_clear_step_change(self):
        """With a clear step at ep 200, detection should be near 200."""
        rewards = _make_step_rewards(n=1000, change_point=200)
        ep = detect_convergence_rolling_mean(rewards, window=50)
        # Should detect convergence within a window-size of the true point
        assert 150 <= ep <= 300, f"Expected ~200, got {ep}"

    def test_already_converged(self):
        """Constant signal: should converge very early (right after warm-up)."""
        rewards = _make_constant_rewards(n=1000)
        ep = detect_convergence_rolling_mean(rewards, window=50)
        assert ep <= 100, f"Expected early convergence, got {ep}"

    def test_gradual_convergence(self):
        """Gradual approach: should detect within reasonable range."""
        rewards = _make_gradual_convergence(n=1000, convergence_ep=150)
        ep = detect_convergence_rolling_mean(rewards, window=50)
        assert ep < 500, f"Expected convergence before ep 500, got {ep}"

    def test_never_converges_returns_high(self):
        """Monotonically increasing: convergence may be detected late or
        at the boundary, but should still return a value."""
        rewards = _make_never_converges(n=1000)
        ep = detect_convergence_rolling_mean(rewards, window=50)
        # The method uses a percentile of the tail, so it will eventually
        # find a match; it should just be fairly late
        assert isinstance(ep, (int, np.integer))


# ───────────────────────────────────────────────────────
# Tests for Method 2: Change-point (t-test)
# ───────────────────────────────────────────────────────


class TestChangePointDetection:
    def test_clear_step_change(self):
        """Should detect the step near episode 200."""
        rewards = _make_step_rewards(n=1000, change_point=200)
        ep = detect_convergence_cusum(rewards, window=50, segment_size=100)
        # The t-test operates on segment boundaries, so expect within
        # ~2 segments of the true change point
        assert 50 <= ep <= 400, f"Expected near 200, got {ep}"

    def test_already_converged(self):
        """Constant signal: should return 0 or very early episode."""
        rewards = _make_constant_rewards(n=1000)
        ep = detect_convergence_cusum(rewards, window=50, segment_size=100)
        # Adjacent segments should never be significantly different
        assert ep <= 150, f"Expected early (or 0), got {ep}"

    def test_small_segment_size(self):
        """Smaller segments give finer resolution."""
        rewards = _make_step_rewards(n=1000, change_point=200)
        ep = detect_convergence_cusum(rewards, window=20, segment_size=50)
        assert 20 <= ep <= 400, f"Expected near 200, got {ep}"

    def test_too_short_returns_zero(self):
        """Series too short for even one segment pair -> return 0."""
        rewards = np.array([0.5] * 50)
        ep = detect_convergence_cusum(rewards, window=50, segment_size=100)
        assert ep == 0


# ───────────────────────────────────────────────────────
# Tests for Method 3: Plateau detection
# ───────────────────────────────────────────────────────


class TestPlateauDetection:
    def test_clear_step_change(self):
        """After the step, the rolling mean slope should flatten."""
        rewards = _make_step_rewards(n=1000, change_point=200)
        ep = detect_convergence_plateau(rewards, window=50, slope_window=100)
        # The plateau should be detected after the step stabilises
        assert 100 <= ep <= 500, f"Expected ~200-300, got {ep}"

    def test_already_converged(self):
        """Constant signal: plateau detected very early."""
        rewards = _make_constant_rewards(n=1000)
        ep = detect_convergence_plateau(rewards, window=50, slope_window=100)
        assert ep <= 250, f"Expected early plateau, got {ep}"

    def test_gradual_convergence(self):
        """Exponential approach: plateau after the signal flattens."""
        rewards = _make_gradual_convergence(n=1000, convergence_ep=150)
        ep = detect_convergence_plateau(rewards, window=50, slope_window=100)
        assert ep < 600, f"Expected plateau before ep 600, got {ep}"

    def test_too_short_returns_n(self):
        """Series too short -> returns len(rewards)."""
        rewards = np.array([0.5] * 50)
        ep = detect_convergence_plateau(rewards, window=50, slope_window=100)
        assert ep == len(rewards)


# ───────────────────────────────────────────────────────
# Cross-method agreement tests
# ───────────────────────────────────────────────────────


class TestCrossMethodAgreement:
    def test_all_methods_agree_on_clear_step(self):
        """All three methods should detect convergence in the same
        approximate region for a clear step change."""
        rewards = _make_step_rewards(n=1000, change_point=200, noise_std=0.1)
        eps = [
            detect_convergence_rolling_mean(rewards, window=50),
            detect_convergence_cusum(rewards, window=50, segment_size=100),
            detect_convergence_plateau(rewards, window=50, slope_window=100),
        ]
        valid = [e for e in eps if 0 < e < 1000]
        assert len(valid) >= 2, f"At least 2 methods should give valid results: {eps}"
        # All valid estimates should be within 300 episodes of each other
        spread = max(valid) - min(valid)
        assert spread <= 300, f"Methods diverge too much: {eps}, spread={spread}"

    def test_all_methods_early_for_constant(self):
        """All methods should return early episodes for a constant signal."""
        rewards = _make_constant_rewards(n=1000)
        eps = [
            detect_convergence_rolling_mean(rewards, window=50),
            detect_convergence_cusum(rewards, window=50, segment_size=100),
            detect_convergence_plateau(rewards, window=50, slope_window=100),
        ]
        # All should be in the first 250 episodes (or 0 for "already converged")
        for e in eps:
            assert e <= 250, f"Expected early convergence, got {eps}"
