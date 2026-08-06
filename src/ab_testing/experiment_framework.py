"""
A/B Testing Experiment Framework
=================================

Provides statistical testing for comparing recommendation strategies.

Uses Welch's t-test (unequal variance), Cohen's d effect size, 95%
confidence intervals, and power analysis.  Only depends on scipy and numpy.

Usage:

    from src.ab_testing.experiment_framework import ABExperiment, ExperimentTracker

    tracker = ExperimentTracker()
    exp = tracker.create_experiment("ts_vs_random", "random", "thompson")
    exp.add_observation("control", reward=0.4)
    exp.add_observation("treatment", reward=0.7)
    ...
    results = exp.get_results()
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# scipy is required for the statistical tests
from scipy import stats as scipy_stats


class ABExperiment:
    """Manages an A/B test between two recommendation strategies."""

    def __init__(
        self,
        name: str,
        control_strategy: str,
        treatment_strategy: str,
        min_sample_size: int = 100,
        significance_level: float = 0.05,
    ):
        """
        Parameters
        ----------
        name : str
            Human-readable experiment name.
        control_strategy : str
            Label for the control (baseline) arm.
        treatment_strategy : str
            Label for the treatment arm.
        min_sample_size : int
            Minimum observations per group before significance can be declared.
        significance_level : float
            Alpha threshold for rejecting the null hypothesis.
        """
        if significance_level <= 0 or significance_level >= 1:
            raise ValueError("significance_level must be in (0, 1)")
        if min_sample_size < 2:
            raise ValueError("min_sample_size must be >= 2")

        self.name = name
        self.control_strategy = control_strategy
        self.treatment_strategy = treatment_strategy
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level

        self._control_rewards: List[float] = []
        self._treatment_rewards: List[float] = []
        self._control_metadata: List[Optional[dict]] = []
        self._treatment_metadata: List[Optional[dict]] = []
        self._created_at = time.time()

    # ── observation recording ────────────────────────────────────────

    def add_observation(
        self, group: str, reward: float, metadata: Optional[dict] = None
    ) -> None:
        """Record a single observation for *group* ('control' or 'treatment')."""
        if group == "control":
            self._control_rewards.append(float(reward))
            self._control_metadata.append(metadata)
        elif group == "treatment":
            self._treatment_rewards.append(float(reward))
            self._treatment_metadata.append(metadata)
        else:
            raise ValueError(f"group must be 'control' or 'treatment', got '{group}'")

    # ── statistical helpers ──────────────────────────────────────────

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Cohen's d (pooled std denominator)."""
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return 0.0
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
        if pooled_std == 0:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled_std)

    @staticmethod
    def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
        """Welch-Satterthwaite degrees of freedom."""
        na, nb = len(a), len(b)
        va = np.var(a, ddof=1) / na
        vb = np.var(b, ddof=1) / nb
        numerator = (va + vb) ** 2
        denominator = va**2 / (na - 1) + vb**2 / (nb - 1)
        if denominator == 0:
            return na + nb - 2
        return numerator / denominator

    @staticmethod
    def _confidence_interval_diff(
        a: np.ndarray, b: np.ndarray, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """95% CI for (mean_treatment - mean_control) using Welch's t."""
        na, nb = len(a), len(b)
        diff = float(np.mean(a) - np.mean(b))
        se = math.sqrt(np.var(a, ddof=1) / na + np.var(b, ddof=1) / nb)
        df = ABExperiment._welch_df(a, b)
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
        return (diff - t_crit * se, diff + t_crit * se)

    @staticmethod
    def _power(
        effect_size: float,
        n_per_group: int,
        alpha: float = 0.05,
    ) -> float:
        """
        Approximate power for a two-sample t-test (equal n).

        Uses non-central t distribution.
        """
        if n_per_group < 2 or effect_size == 0:
            return 0.0
        df = 2 * n_per_group - 2
        ncp = effect_size * math.sqrt(n_per_group / 2)
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
        # Power = P(|T| > t_crit) under non-central t
        power = 1 - (
            scipy_stats.nct.cdf(t_crit, df, ncp) - scipy_stats.nct.cdf(-t_crit, df, ncp)
        )
        return float(np.clip(power, 0, 1))

    # ── public API ───────────────────────────────────────────────────

    def get_results(self) -> dict:
        """
        Compute test statistics and return a results dictionary.

        Returns
        -------
        dict with keys:
            control_mean, treatment_mean, lift_pct,
            t_statistic, p_value, significant,
            confidence_interval_95, sample_sizes,
            effect_size_cohens_d, power
        """
        ctrl = np.asarray(self._control_rewards, dtype=float)
        treat = np.asarray(self._treatment_rewards, dtype=float)
        n_ctrl, n_treat = len(ctrl), len(treat)

        result: Dict = {
            "experiment_name": self.name,
            "control_strategy": self.control_strategy,
            "treatment_strategy": self.treatment_strategy,
            "sample_sizes": {"control": n_ctrl, "treatment": n_treat},
        }

        if n_ctrl < 2 or n_treat < 2:
            # Not enough data for any test
            result.update(
                {
                    "control_mean": float(np.mean(ctrl)) if n_ctrl else None,
                    "treatment_mean": float(np.mean(treat)) if n_treat else None,
                    "lift_pct": None,
                    "t_statistic": None,
                    "p_value": None,
                    "significant": False,
                    "confidence_interval_95": None,
                    "effect_size_cohens_d": None,
                    "power": None,
                }
            )
            return result

        ctrl_mean = float(np.mean(ctrl))
        treat_mean = float(np.mean(treat))

        # Lift percentage (treatment relative to control)
        if ctrl_mean != 0:
            lift_pct = (treat_mean - ctrl_mean) / abs(ctrl_mean) * 100
        else:
            lift_pct = float("inf") if treat_mean != 0 else 0.0

        # Welch's t-test (two-sided)
        t_stat, p_value = scipy_stats.ttest_ind(treat, ctrl, equal_var=False)

        # Significance requires both p < alpha AND min sample sizes met
        enough_data = n_ctrl >= self.min_sample_size and n_treat >= self.min_sample_size
        significant = bool(p_value < self.significance_level and enough_data)

        # 95% CI for (treatment - control)
        ci_lo, ci_hi = self._confidence_interval_diff(
            treat, ctrl, alpha=self.significance_level
        )

        # Cohen's d  (treatment - control)
        d = self._cohens_d(treat, ctrl)

        # Power (use the smaller group size for conservative estimate)
        n_min = min(n_ctrl, n_treat)
        power = self._power(abs(d), n_min, self.significance_level)

        result.update(
            {
                "control_mean": ctrl_mean,
                "treatment_mean": treat_mean,
                "lift_pct": float(lift_pct),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": significant,
                "confidence_interval_95": [float(ci_lo), float(ci_hi)],
                "effect_size_cohens_d": float(d),
                "power": float(power),
            }
        )
        return result

    def is_significant(self) -> bool:
        """Check if we have enough data and the result is statistically significant."""
        return self.get_results()["significant"]

    def get_required_sample_size(self, mde: float = 0.05, power: float = 0.8) -> int:
        """
        Calculate required sample size per group for a minimum detectable effect.

        Parameters
        ----------
        mde : float
            Minimum detectable effect (absolute difference in means).
        power : float
            Desired statistical power (1 - beta).

        Returns
        -------
        int
            Required observations per group.
        """
        if mde <= 0:
            raise ValueError("mde must be positive")
        if not (0 < power < 1):
            raise ValueError("power must be in (0, 1)")

        # Estimate std from existing data, or use 1.0 as default
        all_rewards = self._control_rewards + self._treatment_rewards
        if len(all_rewards) >= 2:
            sigma = float(np.std(all_rewards, ddof=1))
        else:
            sigma = 1.0

        if sigma == 0:
            return self.min_sample_size

        # effect size in Cohen's d units
        d = mde / sigma

        # Required n per group for a two-sample t-test:
        #   n = (z_{1-alpha/2} + z_{power})^2 * 2 / d^2
        z_alpha = scipy_stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = scipy_stats.norm.ppf(power)
        n = math.ceil(2 * ((z_alpha + z_beta) / d) ** 2)
        return max(n, self.min_sample_size)


class ExperimentTracker:
    """Tracks multiple concurrent A/B experiments."""

    def __init__(self):
        self._experiments: Dict[str, ABExperiment] = {}

    def create_experiment(
        self,
        name: str,
        control: str,
        treatment: str,
        **kwargs,
    ) -> ABExperiment:
        """Create and register a new experiment."""
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")
        exp = ABExperiment(
            name=name,
            control_strategy=control,
            treatment_strategy=treatment,
            **kwargs,
        )
        self._experiments[name] = exp
        return exp

    def get_experiment(self, name: str) -> ABExperiment:
        """Retrieve an experiment by name."""
        if name not in self._experiments:
            raise KeyError(f"No experiment named '{name}'")
        return self._experiments[name]

    def list_experiments(self) -> List[str]:
        """Return a list of registered experiment names."""
        return list(self._experiments.keys())

    def get_summary(self) -> dict:
        """Return a summary of all experiments with their results."""
        summary: Dict[str, dict] = {}
        for name, exp in self._experiments.items():
            summary[name] = exp.get_results()
        return summary
