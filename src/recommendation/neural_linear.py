"""
NeuralLinear Contextual Bandit (PyTorch)
========================================

Thompson Sampling over a learned representation: a neural encoder maps the raw
feature vector into a low-dimensional representation, and each action carries an
exact Bayesian linear-regression posterior on top of that representation.

Why this design
---------------
Three options were on the table for the ranker:

*Linear Thompson Sampling* keeps an exact posterior but can only express
reward functions that are linear in the raw features.  The reward here is not:
the dominant term is a squared penalty on the gap between prescribed intensity
and the user's ideal intensity, and that ideal is itself a nonlinear function of
readiness, fatigue and accumulated load.

*A fully Bayesian neural network* (variational, or MC-dropout) can express the
nonlinearity, but its posterior is approximate, and the approximation error
shows up exactly where a bandit is most sensitive — in the tails that drive
exploration.  Under-dispersed posteriors make Thompson Sampling collapse to
greedy.

*NeuralLinear* — the design used here — takes the middle path from Riquelme
et al., "Deep Bayesian Bandits Showdown" (ICLR 2018): learn the representation
by ordinary SGD, then do **exact** conjugate Bayesian inference on the last
layer only.  The nonlinearity is learned; the uncertainty that governs
exploration stays closed-form and calibrated.  That is the property worth
having: exploration is driven by a posterior we can actually trust.

The model
---------
For representation ``z = phi(x) in R^m`` and action ``a``::

    r | z, a  ~  N(w_a^T z, sigma_a^2)
    w_a | sigma_a^2 ~ N(mu_a, sigma_a^2 * A_a^{-1})      (Normal-Inverse-Gamma)
    sigma_a^2 ~ InvGamma(alpha_a, beta_a)

with conjugate updates::

    A_a  <- A_a + z z^T
    b_a  <- b_a + r z
    mu_a  = A_a^{-1} b_a

Acting: draw ``sigma_a^2`` from its Inverse-Gamma posterior, draw ``w_a`` from
the conditional Gaussian, score every allowed action by ``w_a^T z``, take the
argmax.  Both sources of uncertainty — noise scale and coefficients — feed the
exploration, which is what separates this from sampling coefficients alone.

Stale-representation handling
-----------------------------
Every time the encoder is retrained, all previously stored representations are
stale, so the posteriors computed from them are meaningless.  After each
retrain the posteriors are **recomputed from scratch** by re-encoding the whole
replay buffer.  Skipping this step is the classic way to get a NeuralLinear
implementation that silently degrades to noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .bayes_linear import NIGPosteriorSet, batched_ts_probabilities

__all__ = ["NeuralLinearBandit", "NeuralLinearConfig"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NeuralLinearConfig:
    """Hyper-parameters for :class:`NeuralLinearBandit`."""

    hidden_sizes: Tuple[int, ...] = (64, 32)
    repr_dim: int = 32
    lambda_prior: float = 1.0  # prior precision on w_a
    alpha_prior: float = 6.0  # Inverse-Gamma shape
    beta_prior: float = 6.0  # Inverse-Gamma scale
    freeze_encoder: bool = False  # serving mode: posteriors only, never retrain
    train_every: int = 100  # retrain encoder every N observations
    train_epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_train_size: int = 64  # do not retrain before this many observations
    buffer_size: int = 20_000
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    """
    Shared trunk plus per-action reward heads.

    The heads exist only to give the trunk a training signal; at decision time
    they are discarded and the Bayesian linear layer takes over.  Training uses
    a masked loss so each observation only updates the head of the action that
    was actually taken — the reward of the other actions is unobserved, which is
    the whole point of the bandit setting.
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int],
        repr_dim: int,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = feature_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, repr_dim))
        layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.Linear(repr_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def predict_all(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads(self.trunk(x))


# ---------------------------------------------------------------------------
# Bandit
# ---------------------------------------------------------------------------


class NeuralLinearBandit:
    """
    Contextual bandit with a neural representation and exact last-layer
    Bayesian posteriors.

    Parameters
    ----------
    n_actions
        Size of the discrete action space.
    feature_dim
        Dimension of the raw context vector (see
        :data:`src.feature_store.transform.FEATURE_DIM`).
    config
        Hyper-parameters; see :class:`NeuralLinearConfig`.
    seed
        Seeds both the torch generator and the NumPy generator used for
        posterior sampling, so a run is reproducible end to end.
    """

    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        config: Optional[NeuralLinearConfig] = None,
        seed: int = 0,
    ):
        self.cfg = config or NeuralLinearConfig()
        self.n_actions = int(n_actions)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(self.cfg.device)

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.encoder = _Encoder(
            feature_dim=self.feature_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.cfg.hidden_sizes,
            repr_dim=self.cfg.repr_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # Replay buffer of raw (context, action, reward).  Raw, not encoded —
        # the encoder changes, so encoded copies would go stale.
        self._buf_x: List[np.ndarray] = []
        self._buf_a: List[int] = []
        self._buf_r: List[float] = []

        # Exact last-layer posteriors — the same class Linear TS uses, so the
        # two policies differ only in the representation handed to it.
        self.posteriors = NIGPosteriorSet(
            n_actions=self.n_actions,
            dim=self.cfg.repr_dim,
            lambda_prior=self.cfg.lambda_prior,
            alpha_prior=self.cfg.alpha_prior,
            beta_prior=self.cfg.beta_prior,
            seed=seed,
        )
        self._n_observations = 0
        self._n_retrains = 0

    # -- posterior bookkeeping -------------------------------------------

    @property
    def action_counts(self) -> np.ndarray:
        return self.posteriors.counts

    def _rebuild_posteriors(self) -> None:
        """
        Recompute every posterior from the buffer under the current encoder.

        Called after each encoder retrain.  Skipping it would leave posteriors
        that were accumulated in a representation space that no longer exists —
        the classic silent failure mode of NeuralLinear implementations.
        """
        self.posteriors.reset()
        if not self._buf_x:
            return
        Z = self._encode_numpy(np.asarray(self._buf_x, dtype=np.float32))
        for z, a, r in zip(Z, self._buf_a, self._buf_r):
            self.posteriors.update(a, z, r)

    # -- encoding ---------------------------------------------------------

    def _encode_numpy(self, x: np.ndarray) -> np.ndarray:
        """Encode a batch of raw contexts into representations."""
        self.encoder.eval()
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            z = self.encoder(t).cpu().numpy().astype(np.float64)
        return z

    def encode(self, context: np.ndarray) -> np.ndarray:
        """Encode a single raw context vector."""
        return self._encode_numpy(np.asarray(context, dtype=np.float32))[0]

    # -- acting -----------------------------------------------------------

    def select_action(
        self, context: np.ndarray, allowed_actions: Optional[Sequence[int]] = None
    ) -> int:
        """
        Thompson Sampling: one posterior draw per allowed action, take the
        argmax.

        ``allowed_actions`` is the output of the safety gate.  Restricting the
        argmax to that set — rather than masking afterwards — is what keeps
        exploration inside the physiologically feasible region.
        """
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = list(range(self.n_actions))

        z = self.encode(context)
        best_action, best_score = int(allowed_actions[0]), -np.inf
        for aid in allowed_actions:
            w = self.posteriors.sample_weights(int(aid))
            score = float(w @ z)
            if score > best_score:
                best_action, best_score = int(aid), score
        return best_action

    def expected_rewards(
        self, context: np.ndarray, allowed_actions: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """
        Posterior-mean reward estimate for every action.

        Used by the Direct Method / Doubly Robust off-policy estimators as the
        reward model, and by the benchmark's AUC computation.
        """
        z = self.encode(context)
        scores = np.full(self.n_actions, -np.inf)
        ids = (
            range(self.n_actions)
            if allowed_actions is None
            else [int(a) for a in allowed_actions]
        )
        for aid in ids:
            scores[aid] = self.posteriors.mean_reward(aid, z)
        return scores

    def action_probabilities(
        self,
        context: np.ndarray,
        allowed_actions: Optional[Sequence[int]] = None,
        n_samples: int = 200,
    ) -> np.ndarray:
        """
        Monte-Carlo estimate of the Thompson Sampling action distribution.

        Thompson Sampling has no closed-form action probability, but off-policy
        evaluation of *this* policy as a target needs ``pi(a|x)``.  Estimating
        it by repeated posterior sampling is the standard workaround; 200 draws
        is enough for the propensities used downstream.
        """
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = list(range(self.n_actions))
        allowed = [int(a) for a in allowed_actions]

        z = self.encode(context)
        counts = np.zeros(self.n_actions)
        for _ in range(n_samples):
            best_a, best_s = allowed[0], -np.inf
            for aid in allowed:
                s = float(self.posteriors.sample_weights(aid) @ z)
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
        Vectorised ``pi(a | x)`` for a whole batch of raw contexts.

        Encodes the batch once, then shares each joint posterior draw across
        every row — the same trick as Linear TS, and the reason evaluating this
        policy against a 20k-row log takes seconds rather than hours.
        """
        Z = self._encode_numpy(np.asarray(contexts, dtype=np.float32))
        return batched_ts_probabilities(self.posteriors, Z, masks, n_samples)

    # -- learning ---------------------------------------------------------

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Record one observation and update the posterior (and maybe the encoder)."""
        x = np.asarray(context, dtype=np.float64).ravel()
        if x.shape[0] != self.feature_dim:
            raise ValueError(
                f"context has dim {x.shape[0]}, expected {self.feature_dim}"
            )

        self._buf_x.append(x)
        self._buf_a.append(int(action))
        self._buf_r.append(float(reward))
        if len(self._buf_x) > self.cfg.buffer_size:
            self._buf_x.pop(0)
            self._buf_a.pop(0)
            self._buf_r.pop(0)

        self._n_observations += 1

        z = self.encode(x)
        self.posteriors.update(int(action), z, float(reward))

        # Online serving updates the posteriors and nothing else.
        #
        # Retraining the encoder mid-flight would invalidate every stored
        # posterior and force a rebuild from the replay buffer — and a process
        # that started from a checkpoint has an *empty* buffer, so the rebuild
        # would silently discard everything the model had learned before it was
        # saved. (This is not hypothetical: scripts/run_kafka_loop.py caught
        # exactly that, as a posterior count that went backwards.)
        #
        # So the split is: encoder learned offline in batch, where the full
        # history is available; posteriors updated online in closed form, where
        # they are cheap and exact.
        if self.cfg.freeze_encoder:
            return

        if (
            self._n_observations % self.cfg.train_every == 0
            and len(self._buf_x) >= self.cfg.min_train_size
        ):
            self._train_encoder()
            # Representations just changed — every stored posterior is stale.
            self._rebuild_posteriors()

    def _train_encoder(self) -> None:
        """Fit the trunk + heads on the replay buffer with a masked MSE loss."""
        X = torch.as_tensor(
            np.asarray(self._buf_x), dtype=torch.float32, device=self.device
        )
        A = torch.as_tensor(
            np.asarray(self._buf_a), dtype=torch.long, device=self.device
        )
        R = torch.as_tensor(
            np.asarray(self._buf_r), dtype=torch.float32, device=self.device
        )

        n = X.shape[0]
        self.encoder.train()
        for _ in range(self.cfg.train_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                xb, ab, rb = X[idx], A[idx], R[idx]

                preds = self.encoder.predict_all(xb)
                # Only the taken action's head is supervised.
                taken = preds.gather(1, ab.unsqueeze(1)).squeeze(1)
                loss = torch.mean((taken - rb) ** 2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.encoder.eval()
        self._n_retrains += 1

    # -- introspection ----------------------------------------------------

    def statistics(self) -> dict:
        """Diagnostics for logging and tests."""
        return {
            "n_observations": self._n_observations,
            "n_retrains": self._n_retrains,
            "buffer_size": len(self._buf_x),
            "action_counts": self.action_counts.tolist(),
            "posterior_noise_scale": self.posteriors.noise_scale().tolist(),
        }

    def state_dict(self) -> dict:
        """Serialise enough to restore the policy for serving."""
        p = self.posteriors
        return {
            "encoder": self.encoder.state_dict(),
            "A": p.A.copy(),
            "b": p.b.copy(),
            "mu": p.mu.copy(),
            "alpha": p.alpha.copy(),
            "beta": p.beta.copy(),
            "counts": p.counts.copy(),
            "config": self.cfg,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from :meth:`state_dict`."""
        self.encoder.load_state_dict(state["encoder"])
        p = self.posteriors
        p.A = np.asarray(state["A"])
        p.b = np.asarray(state["b"])
        p.mu = np.asarray(state["mu"])
        p.alpha = np.asarray(state["alpha"])
        p.beta = np.asarray(state["beta"])
        p.counts = np.asarray(state["counts"])
        p.A_inv = np.array(
            [
                np.linalg.inv(p.A[a] + 1e-8 * np.eye(self.cfg.repr_dim))
                for a in range(self.n_actions)
            ]
        )
        self.encoder.eval()
