"""
Policy Wrappers
===============

A uniform interface over every strategy the benchmark compares, so the runner
can treat them identically and so off-policy evaluation can ask any of them for
action propensities.

The interface every policy implements:

``act(features, allowed) -> int``
    Choose one action from the safety-gated set.

``propensities(features, allowed) -> np.ndarray``
    ``pi(a | x)`` over the full action space, zero outside ``allowed``.  Exact
    where a closed form exists (uniform, deterministic, epsilon-greedy),
    Monte-Carlo estimated for Thompson Sampling.  This is what makes the
    logged data usable for inverse-propensity estimation.

``update(features, action, reward) -> None``
    Learn from one observation.  A no-op for the fixed baselines.

``scores(features, allowed) -> np.ndarray``
    Per-action preference, used for ranking-quality metrics.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.recommendation.action_space import ActionSpace
from src.recommendation.contextual_bandits import ContextualBandit
from src.recommendation.linear_ts import LinearThompsonSampling
from src.recommendation.neural_linear import NeuralLinearBandit, NeuralLinearConfig

__all__ = [
    "Policy",
    "RandomPolicy",
    "RulePolicy",
    "EpsilonGreedyPolicy",
    "BetaThompsonPolicy",
    "LinTSPolicy",
    "NeuralLinearPolicy",
    "build_policy",
    "POLICY_REGISTRY",
]


class Policy:
    """Base class; subclasses override :meth:`act` and optionally :meth:`update`."""

    name: str = "policy"
    key: str = "policy"

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)

    def act(
        self, features: np.ndarray, allowed: Sequence[int]
    ) -> int:  # pragma: no cover
        raise NotImplementedError

    def propensities(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        """Default: deterministic policies put all mass on the chosen action."""
        p = np.zeros(self.n_actions)
        p[self.act(features, allowed)] = 1.0
        return p

    def propensities_batch(
        self, contexts: np.ndarray, masks: np.ndarray, n_samples: int = 200
    ) -> np.ndarray:
        """
        ``pi(a|x)`` for a whole batch, shape ``(n, n_actions)``.

        The default loops row by row; Thompson Sampling policies override it
        with a vectorised version, because the naive loop costs one posterior
        draw per (row, action, sample) and does not finish on a real log.
        """
        n = contexts.shape[0]
        out = np.zeros((n, self.n_actions))
        for i in range(n):
            allowed = np.flatnonzero(masks[i])
            out[i] = self.propensities(contexts[i], allowed)
        return out

    def update(self, features: np.ndarray, action: int, reward: float) -> None:
        return None

    def scores(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        p = np.full(self.n_actions, -np.inf)
        for a in allowed:
            p[int(a)] = 0.0
        return p

    def statistics(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Fixed baselines
# ---------------------------------------------------------------------------


class RandomPolicy(Policy):
    """Uniform over the safety-gated set.  Propensities are exact."""

    name = "Random Baseline"
    key = "random"

    def __init__(self, n_actions: int, seed: int = 0):
        super().__init__(n_actions)
        self.rng = np.random.default_rng(seed)

    def act(self, features: np.ndarray, allowed: Sequence[int]) -> int:
        return int(self.rng.choice(list(allowed)))

    def propensities(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        p = np.zeros(self.n_actions)
        if len(allowed) == 0:
            return p
        p[[int(a) for a in allowed]] = 1.0 / len(allowed)
        return p

    def propensities_batch(
        self, contexts: np.ndarray, masks: np.ndarray, n_samples: int = 200
    ) -> np.ndarray:
        counts = masks.sum(axis=1, keepdims=True).astype(float)
        return np.where(masks, 1.0 / np.maximum(counts, 1.0), 0.0)

    def scores(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        s = np.full(self.n_actions, -np.inf)
        for a in allowed:
            s[int(a)] = 1.0 / max(len(allowed), 1)
        return s


class RulePolicy(Policy):
    """
    The hand-crafted heuristic the RL policy is meant to replace.

    Deterministic given a state, which is exactly why it can never discover
    that a particular user does better on cardio than strength, and why its
    action coverage is a small fraction of the space.
    """

    name = "Rule-based Heuristic"
    key = "rule_based"

    def __init__(self, n_actions: int, action_space: Optional[ActionSpace] = None):
        super().__init__(n_actions)
        self.space = action_space or ActionSpace()

    def act(self, features: np.ndarray, allowed: Sequence[int]) -> int:
        from src.feature_store.transform import FEATURE_NAMES

        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        readiness = float(features[idx["readiness"]]) * 100.0
        fatigue = float(features[idx["fatigue"]]) * 10.0
        days_since = float(features[idx["days_since_training"]]) * 7.0
        sleep_h = float(features[idx["sleep_hours"]]) * 3.0 + 7.5

        allowed_list = [int(a) for a in allowed]

        if readiness < 40.0 or sleep_h < 5.0:
            return 0 if 0 in allowed_list else allowed_list[0]

        if fatigue >= 7.0:
            recovery = [
                a
                for a in allowed_list
                if self.space.get_action(a).workout_type == "RECOVERY"
            ]
            if recovery:
                return recovery[0]

        if days_since >= 3.0:
            medium = [
                a
                for a in allowed_list
                if self.space.get_action(a).intensity == "MEDIUM"
            ]
            if medium:
                return medium[0]

        low = [a for a in allowed_list if self.space.get_action(a).intensity == "LOW"]
        if low:
            return low[0]

        return allowed_list[0]

    def scores(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        s = np.zeros(self.n_actions)
        s[self.act(features, allowed)] = 1.0
        return s


class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy over a wrapped policy's scores.

    Its purpose is data collection, not performance: it has a known, strictly
    positive propensity on every allowed action, which is the condition
    inverse-propensity estimators need in order to be unbiased.  A logging
    policy that ever assigns probability zero to an action the target policy
    would take makes the target's value unidentifiable from the logs.
    """

    name = "Epsilon-Greedy Logger"
    key = "epsilon_greedy"

    def __init__(self, base: Policy, epsilon: float = 0.3, seed: int = 0):
        super().__init__(base.n_actions)
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        self.base = base
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

    def act(self, features: np.ndarray, allowed: Sequence[int]) -> int:
        p = self.propensities(features, allowed)
        return int(self.rng.choice(self.n_actions, p=p))

    def propensities(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        p = np.zeros(self.n_actions)
        allowed_list = [int(a) for a in allowed]
        if not allowed_list:
            return p
        p[allowed_list] = self.epsilon / len(allowed_list)
        s = self.base.scores(features, allowed_list)
        greedy = int(max(allowed_list, key=lambda a: s[a]))
        p[greedy] += 1.0 - self.epsilon
        return p

    def update(self, features: np.ndarray, action: int, reward: float) -> None:
        self.base.update(features, action, reward)

    def scores(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        return self.base.scores(features, allowed)


# ---------------------------------------------------------------------------
# Learning policies
# ---------------------------------------------------------------------------


class BetaThompsonPolicy(Policy):
    """
    The project's original Beta-Bernoulli Thompson Sampling.

    Kept in the benchmark as the *context-free* control.  It maintains one
    Beta posterior per action and ignores the feature vector entirely, so it
    can learn which actions are good on average but cannot learn which action
    is good *for this user, today*.  The gap between this row and the LinTS row
    is the value of conditioning on context at all.
    """

    name = "Thompson Sampling (context-free)"
    key = "beta_ts"

    def __init__(
        self, n_actions: int, action_space: Optional[ActionSpace] = None, seed: int = 0
    ):
        super().__init__(n_actions)
        space = action_space or ActionSpace()
        self.bandit = ContextualBandit(space)
        np.random.seed(seed)  # ContextualBandit samples from the global RNG

    def act(self, features: np.ndarray, allowed: Sequence[int]) -> int:
        return int(self.bandit.select_action(features, [int(a) for a in allowed]))

    def propensities(
        self, features: np.ndarray, allowed: Sequence[int], n_samples: int = 200
    ) -> np.ndarray:
        counts = np.zeros(self.n_actions)
        for _ in range(n_samples):
            counts[self.act(features, allowed)] += 1
        return counts / float(n_samples)

    def update(self, features: np.ndarray, action: int, reward: float) -> None:
        self.bandit.update(int(action), float(reward))

    def scores(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        s = np.full(self.n_actions, -np.inf)
        for a in allowed:
            a = int(a)
            s[a] = self.bandit.alpha[a] / (self.bandit.alpha[a] + self.bandit.beta[a])
        return s

    def statistics(self) -> dict:
        return {"action_counts": self.bandit.action_counts.tolist()}


class LinTSPolicy(Policy):
    """Linear Thompson Sampling on the raw feature vector."""

    name = "Linear Thompson Sampling"
    key = "lin_ts"

    def __init__(self, n_actions: int, feature_dim: int, seed: int = 0):
        super().__init__(n_actions)
        self.model = LinearThompsonSampling(n_actions, feature_dim, seed=seed)

    def act(self, features: np.ndarray, allowed: Sequence[int]) -> int:
        return int(self.model.select_action(features, [int(a) for a in allowed]))

    def propensities(
        self, features: np.ndarray, allowed: Sequence[int], n_samples: int = 200
    ) -> np.ndarray:
        return self.model.action_probabilities(features, allowed, n_samples=n_samples)

    def propensities_batch(
        self, contexts: np.ndarray, masks: np.ndarray, n_samples: int = 200
    ) -> np.ndarray:
        return self.model.action_probabilities_batch(
            contexts, masks, n_samples=n_samples
        )

    def update(self, features: np.ndarray, action: int, reward: float) -> None:
        self.model.update(features, int(action), float(reward))

    def scores(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        return self.model.expected_rewards(features, allowed)

    def statistics(self) -> dict:
        return self.model.statistics()


class NeuralLinearPolicy(Policy):
    """NeuralLinear: Thompson Sampling on a learned representation."""

    name = "NeuralLinear (PyTorch)"
    key = "neural_linear"

    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        seed: int = 0,
        config: Optional[NeuralLinearConfig] = None,
    ):
        super().__init__(n_actions)
        self.model = NeuralLinearBandit(
            n_actions, feature_dim, config=config, seed=seed
        )

    def act(self, features: np.ndarray, allowed: Sequence[int]) -> int:
        return int(self.model.select_action(features, [int(a) for a in allowed]))

    def propensities(
        self, features: np.ndarray, allowed: Sequence[int], n_samples: int = 200
    ) -> np.ndarray:
        return self.model.action_probabilities(features, allowed, n_samples=n_samples)

    def propensities_batch(
        self, contexts: np.ndarray, masks: np.ndarray, n_samples: int = 200
    ) -> np.ndarray:
        return self.model.action_probabilities_batch(
            contexts, masks, n_samples=n_samples
        )

    def update(self, features: np.ndarray, action: int, reward: float) -> None:
        self.model.update(features, int(action), float(reward))

    def scores(self, features: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
        return self.model.expected_rewards(features, allowed)

    def statistics(self) -> dict:
        return self.model.statistics()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POLICY_REGISTRY = {
    "random": RandomPolicy,
    "rule_based": RulePolicy,
    "beta_ts": BetaThompsonPolicy,
    "lin_ts": LinTSPolicy,
    "neural_linear": NeuralLinearPolicy,
}


def build_policy(key: str, n_actions: int, feature_dim: int, seed: int = 0) -> Policy:
    """Construct a policy by registry key with the right constructor signature."""
    if key == "random":
        return RandomPolicy(n_actions, seed=seed)
    if key == "rule_based":
        return RulePolicy(n_actions)
    if key == "beta_ts":
        return BetaThompsonPolicy(n_actions, seed=seed)
    if key == "lin_ts":
        return LinTSPolicy(n_actions, feature_dim, seed=seed)
    if key == "neural_linear":
        return NeuralLinearPolicy(n_actions, feature_dim, seed=seed)
    raise KeyError(f"unknown policy key: {key!r}")
