"""
Bayesian Linear Posteriors for Thompson Sampling
================================================

A set of independent Normal-Inverse-Gamma posteriors, one per action, over a
shared representation.

This is deliberately factored out of the policies that use it.  Both
:class:`~src.recommendation.linear_ts.LinearThompsonSampling` and
:class:`~src.recommendation.neural_linear.NeuralLinearBandit` sit on top of the
*same* posterior code — the only thing that differs between them is the
representation fed in (raw features vs. a learned encoding).  Keeping the
inference identical is what makes the benchmark comparison between the two an
actual ablation of the representation rather than a comparison of two unrelated
implementations.

The model, per action ``a``, for representation ``z`` and reward ``r``::

    r | z, w_a, sigma_a^2  ~  N(w_a^T z, sigma_a^2)
    w_a | sigma_a^2        ~  N(mu_a, sigma_a^2 A_a^{-1})
    sigma_a^2              ~  InvGamma(alpha_a, beta_a)

Conjugate updates are closed-form::

    A_a <- A_a + z z^T
    b_a <- b_a + r z
    mu_a = A_a^{-1} b_a
    alpha_a <- alpha_a + 1/2
    beta_a  <- beta_a + 1/2 (r^2 + mu_old^T A_old mu_old - mu_new^T A_new mu_new)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["NIGPosteriorSet", "batched_ts_probabilities"]


class NIGPosteriorSet:
    """
    Independent Normal-Inverse-Gamma posteriors, one per action.

    Parameters
    ----------
    n_actions
        Number of arms.
    dim
        Dimension of the representation the posteriors live on.
    lambda_prior
        Prior precision on the coefficients.  Larger = stronger shrinkage
        toward zero and less initial exploration.
    alpha_prior, beta_prior
        Inverse-Gamma prior on the observation-noise variance.  The prior mean
        is ``beta / (alpha - 1)``, so the defaults encode "noise variance is
        around 1.2, with moderate confidence".
    seed
        Seeds the sampler used by :meth:`sample_weights`.
    """

    def __init__(
        self,
        n_actions: int,
        dim: int,
        lambda_prior: float = 1.0,
        alpha_prior: float = 6.0,
        beta_prior: float = 6.0,
        seed: int = 0,
    ):
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if alpha_prior <= 1.0:
            # alpha <= 1 leaves the Inverse-Gamma with undefined mean, which
            # makes the sampled noise scale wander without bound.
            raise ValueError("alpha_prior must be > 1 for a defined posterior mean")

        self.n_actions = int(n_actions)
        self.dim = int(dim)
        self.lambda_prior = float(lambda_prior)
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Return every posterior to the prior."""
        eye = np.eye(self.dim)
        self.A = np.array([self.lambda_prior * eye for _ in range(self.n_actions)])
        self.A_inv = np.array([eye / self.lambda_prior for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, self.dim))
        self.mu = np.zeros((self.n_actions, self.dim))
        self.alpha = np.full(self.n_actions, self.alpha_prior, dtype=float)
        self.beta = np.full(self.n_actions, self.beta_prior, dtype=float)
        self.counts = np.zeros(self.n_actions, dtype=int)

    # -- inference --------------------------------------------------------

    def update(self, action: int, z: np.ndarray, reward: float) -> None:
        """Fold one observation into action ``action``'s posterior."""
        a = int(action)
        z = np.asarray(z, dtype=np.float64).ravel()
        if z.shape[0] != self.dim:
            raise ValueError(
                f"representation has dim {z.shape[0]}, expected {self.dim}"
            )

        A_old = self.A[a].copy()
        mu_old = self.mu[a].copy()

        self.A[a] = A_old + np.outer(z, z)
        self.b[a] = self.b[a] + reward * z
        # Rank-1 update, so Sherman-Morrison is exact and avoids a full inverse.
        self.A_inv[a] = _sherman_morrison(self.A_inv[a], z)
        self.mu[a] = self.A_inv[a] @ self.b[a]

        self.alpha[a] += 0.5
        delta = 0.5 * (
            reward**2
            + float(mu_old @ A_old @ mu_old)
            - float(self.mu[a] @ self.A[a] @ self.mu[a])
        )
        # Numerically the bracket is non-negative; clamp guards float error.
        self.beta[a] += max(delta, 0.0)
        self.counts[a] += 1

    def sample_weights(self, action: int) -> np.ndarray:
        """Draw ``w_a`` from the joint Normal-Inverse-Gamma posterior."""
        a = int(action)
        # sigma^2 ~ InvGamma(alpha, beta)  <=>  beta / Gamma(alpha, 1)
        sigma2 = self.beta[a] / self.rng.gamma(shape=self.alpha[a], scale=1.0)
        cov = sigma2 * self.A_inv[a]
        cov = 0.5 * (cov + cov.T)  # repeated rank-1 updates drift asymmetric
        return self.rng.multivariate_normal(self.mu[a], cov, method="eigh")

    def sample_weights_all(self) -> np.ndarray:
        """
        One posterior draw for *every* action at once, shape ``(n_actions, dim)``.

        Used by the batched propensity estimator: scoring a whole log against a
        single joint draw turns the Monte-Carlo estimate of ``pi(a|x)`` into a
        couple of matrix products instead of one multivariate-normal draw per
        (row, action, sample), which is the difference between seconds and
        hours on a 20k-row log.
        """
        return np.vstack([self.sample_weights(a) for a in range(self.n_actions)])

    def mean_reward(self, action: int, z: np.ndarray) -> float:
        """Posterior-mean predicted reward — the greedy (non-exploring) score."""
        return float(self.mu[int(action)] @ np.asarray(z, dtype=np.float64).ravel())

    def noise_scale(self) -> np.ndarray:
        """Posterior mean of ``sigma_a^2`` for each action."""
        return self.beta / (self.alpha - 1.0)


def batched_ts_probabilities(
    posteriors: NIGPosteriorSet,
    representations: np.ndarray,
    masks: Optional[np.ndarray] = None,
    n_samples: int = 200,
) -> np.ndarray:
    """
    Monte-Carlo estimate of the Thompson Sampling action distribution for a
    batch of representations.

    Parameters
    ----------
    posteriors
        The posterior set to sample from.
    representations
        ``(n, dim)`` representations (raw features for LinTS, encoded for
        NeuralLinear).
    masks
        ``(n, n_actions)`` boolean; ``False`` entries are excluded from the
        argmax, mirroring the safety gate at decision time.
    n_samples
        Number of joint posterior draws.

    Returns
    -------
    np.ndarray
        ``(n, n_actions)`` estimated ``pi(a | x)``; each row sums to 1.

    Notes
    -----
    Thompson Sampling has no closed-form action distribution, so off-policy
    evaluation of it as a *target* policy needs this estimate.  The resolution
    is ``1 / n_samples``: an action whose true probability is below that will
    often be estimated as exactly zero, which for a target policy is harmless
    (it contributes nothing to the value) but would be fatal if this were used
    for *logging* propensities — hence logging uses policies with exact,
    strictly positive closed-form propensities instead.
    """
    Z = np.asarray(representations, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[None, :]
    n = Z.shape[0]
    k = posteriors.n_actions

    counts = np.zeros((n, k))
    neg_inf_mask = None
    if masks is not None:
        neg_inf_mask = ~np.asarray(masks, dtype=bool)

    for _ in range(n_samples):
        W = posteriors.sample_weights_all()  # (k, dim)
        scores = Z @ W.T  # (n, k)
        if neg_inf_mask is not None:
            scores = np.where(neg_inf_mask, -np.inf, scores)
        best = np.argmax(scores, axis=1)
        counts[np.arange(n), best] += 1.0

    return counts / float(n_samples)


def _sherman_morrison(A_inv: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Exact inverse update for ``A + z z^T`` given ``A^{-1}``.

    O(d^2) instead of the O(d^3) a fresh inverse would cost, which matters
    because this runs on every single observation.
    """
    Az = A_inv @ z
    denom = 1.0 + float(z @ Az)
    if abs(denom) < 1e-12:
        # Degenerate; fall back to a regularised explicit inverse.
        return np.linalg.inv(
            np.linalg.inv(A_inv) + np.outer(z, z) + 1e-8 * np.eye(len(z))
        )
    return A_inv - np.outer(Az, Az) / denom
