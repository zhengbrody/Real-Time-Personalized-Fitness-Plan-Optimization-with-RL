"""
Linear Thompson Sampling
========================

Thompson Sampling with an exact Bayesian linear posterior over the *raw*
feature vector.

This is the representation ablation baseline for
:class:`~src.recommendation.neural_linear.NeuralLinearBandit`: identical
posterior machinery (:class:`~src.recommendation.bayes_linear.NIGPosteriorSet`),
identical exploration mechanism, identical safety-gate integration — the only
difference is that the reward model here is linear in the raw features, where
NeuralLinear learns a representation first.

Any performance gap between the two in the benchmark is therefore attributable
to the representation, not to incidental implementation differences.  That is
the point of keeping this class thin.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .bayes_linear import NIGPosteriorSet, batched_ts_probabilities

__all__ = ["LinearThompsonSampling"]


class LinearThompsonSampling:
    """
    Parameters
    ----------
    n_actions
        Size of the discrete action space.
    feature_dim
        Dimension of the raw context vector.
    lambda_prior, alpha_prior, beta_prior
        Passed through to the posterior set.
    seed
        Reproducibility seed for posterior sampling.
    """

    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        lambda_prior: float = 1.0,
        alpha_prior: float = 6.0,
        beta_prior: float = 6.0,
        seed: int = 0,
    ):
        self.n_actions = int(n_actions)
        self.feature_dim = int(feature_dim)
        self.posteriors = NIGPosteriorSet(
            n_actions=self.n_actions,
            dim=self.feature_dim,
            lambda_prior=lambda_prior,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            seed=seed,
        )
        self._n_observations = 0

    # -- acting -----------------------------------------------------------

    def select_action(
        self, context: np.ndarray, allowed_actions: Optional[Sequence[int]] = None
    ) -> int:
        """One posterior draw per allowed action; take the argmax."""
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = list(range(self.n_actions))

        x = np.asarray(context, dtype=np.float64).ravel()
        best_action, best_score = int(allowed_actions[0]), -np.inf
        for aid in allowed_actions:
            score = float(self.posteriors.sample_weights(int(aid)) @ x)
            if score > best_score:
                best_action, best_score = int(aid), score
        return best_action

    def expected_rewards(
        self, context: np.ndarray, allowed_actions: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Posterior-mean reward for every action (``-inf`` where disallowed)."""
        x = np.asarray(context, dtype=np.float64).ravel()
        scores = np.full(self.n_actions, -np.inf)
        ids = (
            range(self.n_actions)
            if allowed_actions is None
            else [int(a) for a in allowed_actions]
        )
        for aid in ids:
            scores[aid] = self.posteriors.mean_reward(aid, x)
        return scores

    def action_probabilities(
        self,
        context: np.ndarray,
        allowed_actions: Optional[Sequence[int]] = None,
        n_samples: int = 200,
    ) -> np.ndarray:
        """
        Monte-Carlo estimate of ``pi(a | x)``.

        Thompson Sampling's action distribution has no closed form; off-policy
        evaluation needs it when this policy is the *target*, so it is
        estimated by repeated posterior sampling.
        """
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = list(range(self.n_actions))
        allowed = [int(a) for a in allowed_actions]

        x = np.asarray(context, dtype=np.float64).ravel()
        counts = np.zeros(self.n_actions)
        for _ in range(n_samples):
            best_a, best_s = allowed[0], -np.inf
            for aid in allowed:
                s = float(self.posteriors.sample_weights(aid) @ x)
                if s > best_s:
                    best_a, best_s = aid, s
            counts[best_a] += 1
        return counts / float(n_samples)

    def action_probabilities_batch(
        self,
        contexts: np.ndarray,
        masks: Optional[np.ndarray] = None,
        n_samples: int = 200,
    ) -> np.ndarray:
        """
        Vectorised ``pi(a | x)`` for a whole batch of contexts.

        Draws one joint set of posterior weights per Monte-Carlo sample and
        scores every context against it.  Each row's estimate is still an
        unbiased average over ``n_samples`` posterior draws; the draws are
        merely shared across rows, which is what makes it fast enough to run on
        a full log.
        """
        return batched_ts_probabilities(
            self.posteriors, np.asarray(contexts, dtype=np.float64), masks, n_samples
        )

    # -- learning ---------------------------------------------------------

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Fold one observation into the posterior."""
        x = np.asarray(context, dtype=np.float64).ravel()
        if x.shape[0] != self.feature_dim:
            raise ValueError(
                f"context has dim {x.shape[0]}, expected {self.feature_dim}"
            )
        self.posteriors.update(int(action), x, float(reward))
        self._n_observations += 1

    # -- introspection ----------------------------------------------------

    def statistics(self) -> dict:
        return {
            "n_observations": self._n_observations,
            "action_counts": self.posteriors.counts.tolist(),
            "posterior_noise_scale": self.posteriors.noise_scale().tolist(),
        }
