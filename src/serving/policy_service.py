"""
Policy Service
==============

Loads the trained NeuralLinear checkpoint and turns a raw request into a
recommendation, going through exactly the same three steps the benchmark does:

    raw signals -> shared transform -> safety gate -> Thompson Sampling

Nothing in this module reimplements any of those steps.  The transform is
imported from ``src.feature_store.transform``, the gate from
``src.safety.action_filter``, the policy from
``src.recommendation.neural_linear``.  That is the whole point: the online path
has no code of its own that could drift away from the offline path.

Startup behaviour
-----------------
If the checkpoint is missing or was trained against a different feature schema,
the service refuses to load it and falls back to an untrained policy, logging
loudly.  Serving a model whose input layout no longer matches the transform is
worse than serving an untrained one — the untrained model is merely bad, the
mismatched one is confidently wrong in ways nobody notices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.feature_store.transform import FEATURE_DIM, FEATURE_NAMES
from src.recommendation.action_space import ActionSpace
from src.recommendation.neural_linear import NeuralLinearBandit, NeuralLinearConfig
from src.safety.action_filter import SafetyDecision, explain, filter_actions

logger = logging.getLogger(__name__)

__all__ = ["PolicyService", "PolicyDecision", "DEFAULT_CHECKPOINT"]

DEFAULT_CHECKPOINT = Path(__file__).parent.parent.parent / "models" / "neural_linear.pt"


@dataclass
class PolicyDecision:
    """One recommendation plus the reasoning behind it."""

    action_id: int
    workout_type: str
    intensity: str
    duration_minutes: int
    description: str
    allowed_action_ids: List[int]
    safety_rules_triggered: List[str]
    safety_explanation: str
    expected_reward: float
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "workout_type": self.workout_type,
            "intensity": self.intensity,
            "duration_minutes": self.duration_minutes,
            "description": self.description,
            "safety": {
                "allowed_action_ids": self.allowed_action_ids,
                "rules_triggered": self.safety_rules_triggered,
                "explanation": self.safety_explanation,
                "n_allowed": len(self.allowed_action_ids),
            },
            "expected_reward": self.expected_reward,
            "model": self.model,
        }


class PolicyService:
    """Serving wrapper around the trained bandit."""

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        action_space: Optional[ActionSpace] = None,
        seed: int = 0,
    ):
        self.action_space = action_space or ActionSpace()
        self.n_actions = self.action_space.get_action_count()
        # Serving freezes the encoder: online feedback moves the last-layer
        # posteriors (closed form, cheap, exact), while the representation is
        # only ever retrained offline where the full replay history exists.
        self.model = NeuralLinearBandit(
            self.n_actions,
            FEATURE_DIM,
            config=NeuralLinearConfig(freeze_encoder=True),
            seed=seed,
        )
        self.model_name = "neural_linear (untrained)"
        self.checkpoint_path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        self._load()

    def _load(self) -> None:
        path = self.checkpoint_path
        if not path.exists():
            logger.warning(
                "no policy checkpoint at %s; serving an untrained policy. "
                "Run scripts/train_policy.py.",
                path,
            )
            return

        try:
            import torch

            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "failed to read checkpoint %s (%s); serving untrained", path, exc
            )
            return

        # Refuse a checkpoint whose feature schema no longer matches the
        # transform: silently feeding it the wrong columns would produce
        # confident nonsense.
        ckpt_dim = int(state.get("feature_dim", -1))
        ckpt_names = list(state.get("feature_names", []))
        if ckpt_dim != FEATURE_DIM or (ckpt_names and ckpt_names != FEATURE_NAMES):
            logger.error(
                "checkpoint feature schema mismatch (checkpoint dim=%s, current=%s); "
                "refusing to load. Retrain with scripts/train_policy.py.",
                ckpt_dim,
                FEATURE_DIM,
            )
            return

        try:
            self.model.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001
            logger.error("checkpoint load failed (%s); serving untrained", exc)
            return

        trained = state.get("training", {})
        self.model_name = (
            f"neural_linear (trained {trained.get('episodes', '?')} episodes)"
        )
        logger.info("loaded policy checkpoint from %s", path)

    # -- inference --------------------------------------------------------

    def recommend(
        self,
        raw_state: Mapping[str, Any],
        features: Optional[np.ndarray] = None,
        explore: bool = True,
    ) -> PolicyDecision:
        """
        Produce one recommendation.

        Parameters
        ----------
        raw_state
            Raw signal dict — the safety gate reads it directly, in raw units,
            rather than trying to invert normalised features.
        features
            Pre-built feature vector (from :class:`FeatureService`).  Built here
            if omitted.
        explore
            ``True`` samples from the posterior (Thompson Sampling).  ``False``
            takes the posterior mean — used for deterministic replay and tests,
            never for live traffic, since a greedy bandit stops learning.
        """
        if features is None:
            from src.feature_store.transform import transform

            features = transform(raw_state)

        decision: SafetyDecision = filter_actions(raw_state, self.action_space)
        allowed = decision.allowed_action_ids

        if explore:
            action_id = int(self.model.select_action(features, allowed))
        else:
            scores = self.model.expected_rewards(features, allowed)
            action_id = int(np.argmax(scores))

        expected = float(self.model.expected_rewards(features, allowed)[action_id])
        action = self.action_space.get_action(action_id)

        return PolicyDecision(
            action_id=action_id,
            workout_type=action.workout_type,
            intensity=action.intensity,
            duration_minutes=action.duration_minutes,
            description=action.description,
            allowed_action_ids=allowed,
            safety_rules_triggered=decision.triggered_rules,
            safety_explanation=explain(decision),
            expected_reward=expected,
            model=self.model_name,
        )

    def update(self, features: np.ndarray, action_id: int, reward: float) -> None:
        """Fold live feedback back into the posterior."""
        self.model.update(features, int(action_id), float(reward))

    def stats(self) -> Dict[str, Any]:
        return {"model": self.model_name, **self.model.statistics()}
