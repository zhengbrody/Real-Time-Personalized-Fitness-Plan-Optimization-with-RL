"""
Heterogeneous Fitness Simulator
===============================

A simulated population of training users, used to benchmark recommendation
policies and to generate logged bandit feedback for off-policy evaluation.

Why this exists
---------------
The first version of the benchmark scored policies against a reward that was a
deterministic function of two observable variables (readiness, fatigue).  That
makes the problem degenerate: the best possible policy is a lookup table, there
is nothing to personalise, and a context-free bandit does about as well as a
contextual one.  Any "our RL beats rules" number measured there says very
little.

This simulator fixes three things:

1. **Heterogeneity.**  Each user draws a latent preference vector — how much
   they tolerate intensity, whether they respond better to strength or cardio,
   how much session duration matters to them.  Two users in an identical
   physiological state have different optimal actions.
2. **Partial observability.**  Some of that latent vector is *correlated with*
   observable profile features (training age, stated goal, self-reported
   tolerance) but never exposed directly.  So context genuinely helps, and yet
   a residual irreducible gap remains — as in reality.
3. **Non-stationarity.**  Fitness adapts.  Consistent training raises a user's
   intensity ceiling over weeks; a layoff lowers it.  The optimal action drifts,
   which is precisely the regime where a continually-updating policy should
   beat a static rule set.

Ground truth
------------
:meth:`FitnessSimulator.expected_reward` returns the *noise-free* expected
reward for a (state, action) pair.  Policies never see it.  It is used for two
things only:

- computing regret against the oracle in the benchmark, and
- calibrating off-policy estimators in ``src/evaluation/ope.py`` — because we
  know the true policy value here, we can measure each estimator's bias, which
  is impossible on real logged data.

Everything is driven by an explicit ``numpy.random.Generator`` so runs are
reproducible from a seed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Sequence

import numpy as np

from src.recommendation.action_space import Action, ActionSpace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTENSITY_LEVEL: Dict[str, int] = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

# Training-load cost of one session, used to drive ACWR and fatigue dynamics.
_LOAD_PER_MINUTE: Dict[str, float] = {
    "NONE": 0.0,
    "LOW": 3.0,
    "MEDIUM": 6.0,
    "HIGH": 10.0,
}

_GOALS = ("strength", "endurance", "general")


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    """
    Static per-user attributes.

    The ``observable_*`` fields are exposed to the policy through the feature
    transform.  The ``latent_*`` fields are not — they only enter the reward.
    """

    user_id: int

    # Observable profile (fed to the policy via the feature transform)
    age: float
    bodyweight_kg: float
    training_age_years: float
    goal: str
    stated_intensity_tolerance: float  # self-report, a noisy view of the latent

    # Latent preferences (never observed by any policy)
    latent_intensity_tolerance: float  # 0 = fragile, 1 = thrives on hard work
    latent_type_affinity: float  # -1 = pure cardio person, +1 = pure strength
    latent_duration_pref: float  # preferred session length in minutes
    latent_consistency: float  # propensity to complete what is prescribed
    latent_noise_scale: float  # how noisy this user's feedback is

    # Adaptation dynamics
    adaptation_rate: float  # how fast fitness responds to load


@dataclass
class UserState:
    """Mutable per-user state that evolves day by day."""

    profile: UserProfile
    day: int = 0

    # Physiological trajectory
    fitness_level: float = 0.5  # drifts with training load (non-stationarity)
    baseline_hrv: float = 50.0
    baseline_resting_hr: float = 60.0
    accumulated_fatigue: float = 3.0

    # Rolling history used to derive features honestly
    hrv_history: Deque[float] = field(default_factory=lambda: deque(maxlen=28))
    load_history: Deque[float] = field(default_factory=lambda: deque(maxlen=28))
    completion_history: Deque[int] = field(default_factory=lambda: deque(maxlen=28))
    sleep_history: Deque[float] = field(default_factory=lambda: deque(maxlen=28))

    days_since_training: int = 1
    consecutive_hard_days: int = 0
    streak: int = 0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class FitnessSimulator:
    """
    A population of heterogeneous, adapting simulated users.

    Typical use::

        sim = FitnessSimulator(seed=42)
        user = sim.new_user()
        for day in range(1000):
            state = sim.observe(user)
            allowed = safety_filter(state)
            action_id = policy.select(state, allowed)
            reward = sim.step(user, action_id)
            policy.update(state, action_id, reward)
    """

    def __init__(
        self,
        action_space: Optional[ActionSpace] = None,
        seed: int = 0,
        enable_adaptation: bool = True,
    ):
        self.action_space = action_space or ActionSpace()
        self.rng = np.random.default_rng(seed)
        self.enable_adaptation = enable_adaptation
        self._next_user_id = 0

    # -- population ------------------------------------------------------

    def new_user(self) -> UserState:
        """Draw a fresh user from the population distribution."""
        rng = self.rng
        uid = self._next_user_id
        self._next_user_id += 1

        latent_tolerance = float(np.clip(rng.beta(2.5, 2.5), 0.02, 0.98))
        training_age = float(np.clip(rng.gamma(2.0, 2.0), 0.0, 10.0))

        # Stated tolerance is the latent value seen through self-report noise —
        # informative but not sufficient, which is what makes the problem hard.
        stated = float(np.clip(latent_tolerance + rng.normal(0.0, 0.18), 0.0, 1.0))

        type_affinity = float(np.clip(rng.normal(0.0, 0.6), -1.0, 1.0))
        # Stated goal correlates with, but does not determine, type affinity.
        if type_affinity > 0.35:
            goal = "strength" if rng.random() < 0.75 else "general"
        elif type_affinity < -0.35:
            goal = "endurance" if rng.random() < 0.75 else "general"
        else:
            goal = str(rng.choice(_GOALS))

        profile = UserProfile(
            user_id=uid,
            age=float(np.clip(rng.normal(35.0, 11.0), 18.0, 70.0)),
            bodyweight_kg=float(np.clip(rng.normal(75.0, 14.0), 45.0, 130.0)),
            training_age_years=training_age,
            goal=goal,
            stated_intensity_tolerance=stated,
            latent_intensity_tolerance=latent_tolerance,
            latent_type_affinity=type_affinity,
            latent_duration_pref=float(np.clip(rng.normal(35.0, 10.0), 15.0, 60.0)),
            latent_consistency=float(np.clip(rng.beta(5.0, 2.0), 0.2, 1.0)),
            latent_noise_scale=float(np.clip(rng.gamma(4.0, 0.03), 0.04, 0.30)),
            adaptation_rate=float(np.clip(rng.normal(0.012, 0.004), 0.002, 0.03)),
        )

        state = UserState(
            profile=profile,
            fitness_level=float(np.clip(0.3 + 0.05 * training_age, 0.1, 0.9)),
            baseline_hrv=float(np.clip(rng.normal(55.0, 12.0), 25.0, 95.0)),
            baseline_resting_hr=float(np.clip(rng.normal(58.0, 7.0), 42.0, 80.0)),
        )

        # Warm up the rolling windows so early episodes are not dominated by
        # cold-start defaults.
        for _ in range(14):
            state.hrv_history.append(
                float(np.clip(rng.normal(state.baseline_hrv, 6.0), 20.0, 110.0))
            )
            state.sleep_history.append(float(np.clip(rng.normal(7.2, 0.9), 4.0, 10.0)))
            state.load_history.append(
                float(np.clip(rng.normal(150.0, 70.0), 0.0, 450.0))
            )
            state.completion_history.append(1 if rng.random() < 0.7 else 0)

        return state

    # -- observation -----------------------------------------------------

    def observe(self, user: UserState) -> Dict[str, float]:
        """
        Produce today's raw body-state record.

        The returned dict uses the same field names the real wearable pipeline
        emits, so it can be handed straight to
        :func:`src.feature_store.transform.transform`.
        """
        rng = self.rng
        p = user.profile
        day = user.day

        # Weekly circadian-ish rhythm plus fatigue drag.
        week_phase = (day % 7) / 7.0
        rhythm = 8.0 * np.sin(2.0 * np.pi * week_phase + np.pi)

        hrv_28d = (
            float(np.mean(user.hrv_history)) if user.hrv_history else user.baseline_hrv
        )
        hrv_std = float(np.std(user.hrv_history)) if len(user.hrv_history) > 1 else 6.0

        # Today's HRV: baseline, lifted by fitness, dragged by fatigue.
        hrv = float(
            np.clip(
                rng.normal(
                    user.baseline_hrv
                    + 10.0 * (user.fitness_level - 0.5)
                    - 2.2 * user.accumulated_fatigue
                    + rhythm * 0.4,
                    5.5,
                ),
                20.0,
                110.0,
            )
        )

        sleep_hours = float(np.clip(rng.normal(7.3, 0.85), 4.0, 10.0))
        sleep_score = float(
            np.clip(
                rng.normal(
                    55.0 + 4.0 * sleep_hours - 1.5 * user.accumulated_fatigue, 8.0
                ),
                25.0,
                100.0,
            )
        )

        resting_hr = float(
            np.clip(
                rng.normal(
                    user.baseline_resting_hr
                    - 4.0 * (user.fitness_level - 0.5)
                    + 1.3 * user.accumulated_fatigue,
                    3.5,
                ),
                40.0,
                95.0,
            )
        )

        fatigue = float(
            np.clip(user.accumulated_fatigue + rng.normal(0.0, 0.5), 1.0, 10.0)
        )

        # Readiness is a composite the wearable would report — deliberately a
        # noisy summary, so it does not fully determine the optimal action.
        readiness = float(
            np.clip(
                rng.normal(
                    45.0
                    + 0.45 * (hrv - user.baseline_hrv)
                    + 3.0 * (sleep_hours - 7.0)
                    + 0.25 * (sleep_score - 70.0)
                    - 4.0 * (fatigue - 5.0)
                    + 18.0 * (user.fitness_level - 0.5),
                    7.0,
                ),
                5.0,
                100.0,
            )
        )

        activity_score = float(np.clip(rng.normal(62.0, 16.0), 15.0, 100.0))

        hrv_7d = (
            float(np.mean(list(user.hrv_history)[-7:])) if user.hrv_history else hrv
        )
        load_7d = (
            float(np.sum(list(user.load_history)[-7:])) if user.load_history else 0.0
        )
        load_28d_mean = float(np.mean(user.load_history)) if user.load_history else 0.0
        acute = (
            float(np.mean(list(user.load_history)[-7:])) if user.load_history else 0.0
        )
        chronic = load_28d_mean if load_28d_mean > 1e-6 else 1.0
        acwr = float(np.clip(acute / chronic, 0.0, 3.0))

        sleep_debt = (
            float(np.sum([7.5 - h for h in list(user.sleep_history)[-7:]]))
            if user.sleep_history
            else 0.0
        )

        comp = list(user.completion_history)
        completion_7d = float(np.mean(comp[-7:])) if comp else 0.7
        completion_28d = float(np.mean(comp)) if comp else 0.7
        sessions_7d = float(
            np.sum([1 for load in list(user.load_history)[-7:] if load > 1.0])
        )
        days_active_28d = float(np.sum([1 for load in user.load_history if load > 1.0]))

        return {
            # recovery
            "readiness_score": readiness,
            "sleep_score": sleep_score,
            "sleep_duration_hours": sleep_hours,
            "sleep_debt_7d": sleep_debt,
            "hrv": hrv,
            "hrv_7d_mean": hrv_7d,
            "hrv_28d_mean": hrv_28d,
            "hrv_28d_std": hrv_std,
            "resting_hr": resting_hr,
            "resting_hr_baseline": user.baseline_resting_hr,
            # load
            "fatigue": fatigue,
            "activity_score": activity_score,
            "acwr": acwr,
            "load_7d": load_7d,
            "load_28d_mean": load_28d_mean,
            "days_since_training": float(user.days_since_training),
            "consecutive_hard_days": float(user.consecutive_hard_days),
            # consistency
            "completion_rate_7d": completion_7d,
            "completion_rate_28d": completion_28d,
            "streak": float(user.streak),
            "sessions_7d": sessions_7d,
            "days_active_28d": days_active_28d,
            # temporal
            "day_of_week": float(day % 7),
            "week_phase": week_phase,
            # profile
            "training_age_years": p.training_age_years,
            "age": p.age,
            "goal": p.goal,
            "intensity_tolerance": p.stated_intensity_tolerance,
            "bodyweight_kg": p.bodyweight_kg,
        }

    # -- reward ----------------------------------------------------------

    def ideal_intensity(self, user: UserState, state: Dict[str, float]) -> float:
        """
        The continuous intensity level (0-3) that maximises today's reward.

        Depends on observable physiology *and* on the user's latent tolerance
        and current fitness — which is why a context-free policy cannot track
        it and a static rule set cannot personalise it.
        """
        p = user.profile
        readiness = state["readiness_score"]
        fatigue = state["fatigue"]
        acwr = state["acwr"]

        base = (
            3.2 * (readiness / 100.0)
            - 0.13 * fatigue
            + 1.5 * (p.latent_intensity_tolerance - 0.5)
            + 1.1 * (user.fitness_level - 0.5)
            - 0.55 * max(0.0, acwr - 1.35)
            + 0.62
        )
        return float(np.clip(base, 0.0, 3.0))

    def expected_reward(
        self, user: UserState, state: Dict[str, float], action: Action
    ) -> float:
        """
        Noise-free expected reward for taking ``action`` in ``state``.

        **Ground truth — never exposed to a policy.**  Used for oracle/regret
        computation and for calibrating off-policy estimators.
        """
        p = user.profile
        level = _INTENSITY_LEVEL[action.intensity]
        ideal = self.ideal_intensity(user, state)

        # 1. Intensity match — the dominant term, asymmetric because
        #    overshooting is worse than undershooting.
        gap = level - ideal
        if gap >= 0:
            intensity_term = -0.30 * gap**2
        else:
            intensity_term = -0.17 * gap**2

        # 2. Modality affinity — cardio vs strength, per user.
        if action.workout_type == "STRENGTH":
            type_term = 0.16 * p.latent_type_affinity
        elif action.workout_type == "CARDIO":
            type_term = -0.16 * p.latent_type_affinity
        else:
            type_term = 0.0

        # 3. Duration fit, only meaningful for actual sessions.
        if action.duration_minutes > 0:
            dur_gap = (action.duration_minutes - p.latent_duration_pref) / 30.0
            duration_term = -0.10 * dur_gap**2
        else:
            duration_term = 0.0

        # 4. Rest is valuable when depleted and wasteful when fresh.
        if action.workout_type == "REST":
            rest_term = 0.42 * (1.0 - state["readiness_score"] / 100.0) - 0.16
        else:
            rest_term = 0.0

        # 5. Hard overtraining penalty — the regime the safety gate exists for.
        overtraining = level >= 3 and (
            state["fatigue"] > 7.0 or state["readiness_score"] < 40.0
        )
        overtraining_term = -0.55 if overtraining else 0.0

        # 6. Consistent users complete more of what they are given.
        adherence_term = 0.10 * (p.latent_consistency - 0.6)

        total = (
            0.72
            + intensity_term
            + type_term
            + duration_term
            + rest_term
            + overtraining_term
            + adherence_term
        )
        return float(np.clip(total, -1.0, 1.0))

    def sample_reward(
        self, user: UserState, state: Dict[str, float], action: Action
    ) -> float:
        """Draw an observed reward: expected value plus per-user feedback noise."""
        mean = self.expected_reward(user, state, action)
        noisy = self.rng.normal(mean, user.profile.latent_noise_scale)
        return float(np.clip(noisy, -1.0, 1.0))

    def optimal_action(
        self,
        user: UserState,
        state: Dict[str, float],
        allowed_actions: Optional[Sequence[int]] = None,
    ) -> int:
        """Oracle: the highest-expected-reward action among those allowed."""
        if allowed_actions is None:
            allowed_actions = range(self.action_space.get_action_count())
        best_id, best_value = -1, -np.inf
        for aid in allowed_actions:
            value = self.expected_reward(user, state, self.action_space.get_action(aid))
            if value > best_value:
                best_id, best_value = int(aid), value
        return best_id

    def optimal_value(
        self,
        user: UserState,
        state: Dict[str, float],
        allowed_actions: Optional[Sequence[int]] = None,
    ) -> float:
        """Expected reward of the oracle action — the regret reference point."""
        aid = self.optimal_action(user, state, allowed_actions)
        return self.expected_reward(user, state, self.action_space.get_action(aid))

    # -- transition ------------------------------------------------------

    def step(self, user: UserState, state: Dict[str, float], action_id: int) -> float:
        """
        Apply ``action_id``, advance the user one day, and return the observed
        reward.

        Fitness adapts to accumulated load, so the optimal policy drifts over
        the course of a run.
        """
        action = self.action_space.get_action(action_id)
        reward = self.sample_reward(user, state, action)

        level = _INTENSITY_LEVEL[action.intensity]
        load = _LOAD_PER_MINUTE[action.intensity] * action.duration_minutes

        # Completion is stochastic: harder sessions are skipped more often by
        # less consistent users.
        completion_p = float(
            np.clip(user.profile.latent_consistency - 0.06 * level, 0.05, 0.99)
        )
        completed = 1 if self.rng.random() < completion_p else 0
        realised_load = load * completed

        # Fatigue accumulates with load and decays toward baseline.  The
        # coefficient is set so that sustained maximal load (a 45-min HIGH
        # session every day) settles near fatigue 9, rest settles near 2, and a
        # moderate week sits around 5 — i.e. the whole 1-10 scale is used.
        user.accumulated_fatigue = float(
            np.clip(
                0.86 * user.accumulated_fatigue + 0.0022 * realised_load + 0.28,
                1.0,
                10.0,
            )
        )

        # Fitness adapts to training stimulus.  Detraining is deliberately
        # slower than adaptation (a third of the rate): without that asymmetry
        # a myopic bandit that rests on a bad day pushes the user into a
        # spiral where lower fitness depresses readiness, which justifies more
        # rest — a degenerate attractor that is an artefact of the simulator
        # rather than a property of training.
        if self.enable_adaptation:
            stimulus = (realised_load / 300.0) - 0.32
            if stimulus < 0.0:
                stimulus *= 0.35
            user.fitness_level = float(
                np.clip(
                    user.fitness_level + user.profile.adaptation_rate * stimulus,
                    0.05,
                    0.98,
                )
            )

        # Update rolling history.
        user.hrv_history.append(state["hrv"])
        user.sleep_history.append(state["sleep_duration_hours"])
        user.load_history.append(realised_load)
        user.completion_history.append(completed)

        if realised_load > 1.0:
            user.days_since_training = 0
            user.streak += 1
        else:
            user.days_since_training = min(user.days_since_training + 1, 7)
            user.streak = 0

        if level >= 2 and completed:
            user.consecutive_hard_days = min(user.consecutive_hard_days + 1, 5)
        else:
            user.consecutive_hard_days = 0

        user.day += 1
        return reward
