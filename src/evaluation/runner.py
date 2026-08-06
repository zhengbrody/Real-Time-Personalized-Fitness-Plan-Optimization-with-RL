"""
Benchmark Runner
================

Runs a policy against the simulated population and records everything needed
for both online metrics (reward, regret, coverage) and offline evaluation
(propensities, features, per-action ground truth).

Experimental design
-------------------
Episodes are **interleaved across a population of users**, round-robin: episode
*t* is day ``t // n_users`` for user ``t % n_users``.  This matters.  If a run
were a single user for 1000 consecutive days, a context-free bandit would do
nearly as well as a contextual one — it could just learn that user's average
preferences.  Interleaving many users with different latent traits means the
right action flips from episode to episode, and the only way to track it is to
condition on the context.  That is the regime the product actually operates in.

Every policy is run against a simulator constructed with the same seed, so all
policies see the same user population and the same physiological noise draws.
Trajectories still diverge — the environment is closed-loop, and a policy that
prescribes differently produces different fatigue and fitness downstream — but
that divergence is caused by the policy, which is the thing being measured.

Regret is computed against the **noise-free** oracle value, restricted to the
same safety-gated action set the policy had to choose from.  Charging a policy
for actions the safety gate forbade would measure the gate, not the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.policies import Policy
from src.feature_store.transform import FEATURE_DIM, transform
from src.recommendation.action_space import ActionSpace
from src.safety.action_filter import filter_actions
from src.simulation.fitness_env import FitnessSimulator

__all__ = [
    "StepRecord",
    "run_policy",
    "collect_logged_data",
    "compute_metrics",
    "rolling_mean",
]


@dataclass
class StepRecord:
    """One (user, day) interaction, with everything both evaluators need."""

    episode: int
    user_id: int
    day: int
    features: np.ndarray
    allowed: List[int]
    action: int
    propensity: float
    reward: float
    expected_reward: float
    oracle_action: int
    oracle_value: float
    regret: float
    is_optimal: bool
    is_overtraining: bool
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))


def run_policy(
    policy: Policy,
    n_episodes: int = 1000,
    n_users: int = 50,
    seed: int = 0,
    action_space: Optional[ActionSpace] = None,
    record_propensities: bool = False,
    record_scores: bool = True,
) -> List[StepRecord]:
    """
    Run ``policy`` for ``n_episodes`` interleaved user-days.

    Parameters
    ----------
    record_propensities
        Estimate ``pi(a|x)`` at each step.  Off by default because Thompson
        Sampling needs 200 posterior draws per step to estimate it, which is
        ~100x the cost of acting.  Turn it on when generating logged data for
        off-policy evaluation.
    record_scores
        Keep per-action scores for ranking metrics.  Off for large logging runs
        to save memory.
    """
    space = action_space or ActionSpace()
    sim = FitnessSimulator(action_space=space, seed=seed)
    users = [sim.new_user() for _ in range(n_users)]

    records: List[StepRecord] = []
    for episode in range(n_episodes):
        user = users[episode % n_users]
        raw_state = sim.observe(user)
        features = transform(raw_state)

        decision = filter_actions(raw_state, space)
        allowed = decision.allowed_action_ids

        action = int(policy.act(features, allowed))
        if action not in allowed:
            raise RuntimeError(
                f"{policy.name} returned action {action} outside the safety-gated "
                f"set {allowed} — the gate is not being honoured."
            )

        propensity = 1.0
        if record_propensities:
            probs = policy.propensities(features, allowed)
            propensity = float(probs[action])

        scores = policy.scores(features, allowed) if record_scores else np.zeros(0)

        # Ground truth, computed before the environment advances.
        chosen_action = space.get_action(action)
        expected = sim.expected_reward(user, raw_state, chosen_action)
        oracle_action = sim.optimal_action(user, raw_state, allowed)
        oracle_value = sim.expected_reward(
            user, raw_state, space.get_action(oracle_action)
        )

        reward = sim.step(user, raw_state, action)
        policy.update(features, action, reward)

        records.append(
            StepRecord(
                episode=episode,
                user_id=user.profile.user_id,
                day=user.day - 1,
                features=features,
                allowed=list(allowed),
                action=action,
                propensity=propensity,
                reward=float(reward),
                expected_reward=float(expected),
                oracle_action=int(oracle_action),
                oracle_value=float(oracle_value),
                regret=float(max(oracle_value - expected, 0.0)),
                is_optimal=bool(action == oracle_action),
                is_overtraining=bool(
                    chosen_action.intensity == "HIGH"
                    and (
                        raw_state["fatigue"] > 7.0
                        or raw_state["readiness_score"] < 40.0
                    )
                ),
                scores=scores,
            )
        )

    return records


def collect_logged_data(
    behaviour_policy: Policy,
    n_episodes: int = 5000,
    n_users: int = 100,
    seed: int = 0,
    action_space: Optional[ActionSpace] = None,
    learn: bool = False,
):
    """
    Run a behaviour policy and return a :class:`~src.evaluation.ope.LoggedBatch`.

    This is the offline-evaluation counterpart to :func:`run_policy`.  The
    difference that matters: the propensity ``mu(a|x)`` is recorded **at
    decision time**, from the same distribution the action was drawn from.

    Reconstructing propensities afterwards — re-running the policy on the
    logged context and asking what it would do now — is the single most common
    way to get off-policy evaluation quietly wrong.  A learning policy's
    distribution at replay time is not the one that produced the action, and
    every inverse-propensity estimate built on it is biased by an amount
    nobody can measure.

    Parameters
    ----------
    learn
        Whether the behaviour policy updates as it collects.  Default ``False``:
        a *fixed* logging policy makes the logged data i.i.d. given the context
        stream, which is what the estimators assume.  Set ``True`` to study the
        (harder, more realistic) adaptively-logged case.
    """
    from src.evaluation.ope import LoggedBatch  # local: avoids a circular import

    space = action_space or ActionSpace()
    n_actions = space.get_action_count()
    sim = FitnessSimulator(action_space=space, seed=seed)
    users = [sim.new_user() for _ in range(n_users)]

    contexts = np.zeros((n_episodes, FEATURE_DIM))
    actions = np.zeros(n_episodes, dtype=int)
    rewards = np.zeros(n_episodes)
    propensities = np.zeros(n_episodes)
    masks = np.zeros((n_episodes, n_actions), dtype=bool)
    true_q = np.zeros((n_episodes, n_actions))

    # One generator for the whole run: the action draw must be independent
    # across episodes for the logged data to satisfy the estimators' i.i.d.
    # assumption.
    draw_rng = np.random.default_rng(seed + 7_919)

    for episode in range(n_episodes):
        user = users[episode % n_users]
        raw_state = sim.observe(user)
        features = transform(raw_state)
        allowed = filter_actions(raw_state, space).allowed_action_ids

        probs = behaviour_policy.propensities(features, allowed)
        total = probs.sum()
        if total <= 0:
            raise RuntimeError(
                f"{behaviour_policy.name} produced an empty action distribution"
            )
        probs = probs / total
        action = int(draw_rng.choice(n_actions, p=probs))

        # Ground truth for every action, recorded before the state advances.
        for aid in range(n_actions):
            true_q[episode, aid] = sim.expected_reward(
                user, raw_state, space.get_action(aid)
            )

        reward = sim.step(user, raw_state, action)
        if learn:
            behaviour_policy.update(features, action, reward)

        contexts[episode] = features
        actions[episode] = action
        rewards[episode] = reward
        propensities[episode] = probs[action]
        masks[episode, allowed] = True

    return LoggedBatch(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        propensities=propensities,
        allowed_masks=masks,
        true_q=true_q,
    )


def rolling_mean(values: Sequence[float], window: int = 50) -> List[float]:
    """Trailing rolling mean, same length as the input."""
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def _episode_auc(scores: np.ndarray, allowed: Sequence[int], optimal: int) -> float:
    """
    Rank quality for one episode: does the policy score the oracle action above
    the other legal actions?

    Equivalent to the probability that a randomly chosen non-optimal legal
    action scores below the optimal one, with ties counted as half — i.e. the
    standard rank-based AUC, restricted to the safety-gated set.
    """
    allowed = [int(a) for a in allowed]
    if len(allowed) < 2 or optimal not in allowed or scores.size == 0:
        return float("nan")

    target = scores[optimal]
    others = [scores[a] for a in allowed if a != optimal]
    wins = sum(1.0 for s in others if target > s)
    ties = sum(1.0 for s in others if target == s)
    return float((wins + 0.5 * ties) / len(others))


def compute_metrics(records: Sequence[StepRecord], window: int = 50) -> Dict[str, Any]:
    """Aggregate a run into the metrics reported by the benchmark."""
    if not records:
        raise ValueError("no records to score")

    rewards = np.array([r.reward for r in records], dtype=float)
    regrets = np.array([r.regret for r in records], dtype=float)
    n = len(records)
    tail = max(1, n // 5)

    aucs = [
        _episode_auc(r.scores, r.allowed, r.oracle_action)
        for r in records
        if r.scores.size
    ]
    valid_aucs = [a for a in aucs if not np.isnan(a)]

    action_counts = np.zeros(18, dtype=int)
    for r in records:
        action_counts[r.action] += 1
    used = action_counts > 0
    share = action_counts / float(n)

    return {
        "n_episodes": n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "cumulative_reward": float(np.sum(rewards)),
        # Regret
        "cumulative_regret": float(np.sum(regrets)),
        "mean_regret": float(np.mean(regrets)),
        "final_regret_rate": float(np.mean(regrets[-tail:])),
        "regret_curve": np.cumsum(regrets).tolist(),
        # Decision quality
        "optimal_action_rate": float(np.mean([r.is_optimal for r in records])),
        "overtraining_rate": float(np.mean([r.is_overtraining for r in records])),
        "mean_auc": float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        # Coverage: how much of the action space the policy actually uses.
        # A deterministic rule set collapses onto a handful of actions and can
        # never discover the rest; this is the number that quantifies it.
        "actions_used": int(used.sum()),
        "actions_used_above_1pct": int((share >= 0.01).sum()),
        "action_distribution": action_counts.tolist(),
        # Learning progress
        "first_half_mean_reward": float(np.mean(rewards[: n // 2])),
        "second_half_mean_reward": float(np.mean(rewards[n // 2 :])),
        "learning_improvement": float(
            np.mean(rewards[n // 2 :]) - np.mean(rewards[: n // 2])
        ),
        "rolling_rewards": rolling_mean(rewards, window),
    }


def convergence_episode(records: Sequence[StepRecord], window: int = 50) -> int:
    """
    First episode at which the rolling reward reaches the 25th percentile of
    its own final-100-episode band — a stable-performance threshold rather than
    a peak, so a single lucky episode cannot declare convergence.
    """
    rewards = [r.reward for r in records]
    rolling = rolling_mean(rewards, window)
    target = float(np.percentile(rolling[-100:], 25))
    for i, v in enumerate(rolling):
        if i > window and v >= target:
            return i
    return len(records)


def pct_improvement(
    a: Dict[str, Any], b: Dict[str, Any], key: str = "mean_reward"
) -> float:
    """Percentage improvement of ``a`` over ``b`` on ``key``."""
    base = b[key]
    return (a[key] - base) / max(abs(base), 1e-9) * 100.0
