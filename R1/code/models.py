# models.py
"""
BNN with fully optimised MC Dropout inference for FedUMPC.

Performance architecture:
──────────────────────────────────────────────────────────────────────────
  SLSQP calls _cost() ~50-100× per control step.
  Old code:  each _cost() call → H steps × M serial forward() calls
             = 12 × 20 = 240 Python-level forward passes per cost eval
             = ~12,000 passes per control step  ← SLOW

  New code:  _cost() pre-builds an (H,6) batch, calls predict_horizon()
             ONCE → single (M×H, 6) forward pass via _mc_batch()
             = 1 batched forward per cost eval (GPU/CPU fully utilised)
             Expected speedup: 8-15× on CPU, 20-40× on CUDA

  Additional micro-optimisations:
  • Pre-computed decay weights  λ_h  (numpy array, built once at init)
  • Pre-computed horizon indices (1 + 0.15*h) array
  • Reuse pre-allocated (H,6) numpy buffer inside _cost()
  • torch.no_grad() + self.train() set once, never toggled mid-loop
  • CPU↔GPU copy batched: one torch.tensor() call for whole horizon
──────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import numpy as np
from config import Config


class BNN(nn.Module):
    """
    Bayesian Neural Network with MC Dropout.
    Input:  [s; u] ∈ R^{6}   Output: Δs ∈ R^{4}  (state increment)
    """

    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(6, Config.HIDDEN_DIM)
        self.fc2     = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        self.fc3     = nn.Linear(Config.HIDDEN_DIM, 4)
        self.dropout = nn.Dropout(p=Config.DROPOUT)
        self._M      = Config.MC_SAMPLES
        self.register_buffer(
            'input_scale',
            torch.tensor([22.0, 22.0, 12.0, np.pi,
                          Config.MAX_STEER, Config.MAX_ACCEL],
                         dtype=torch.float32)
        )
        self.register_buffer(
            'delta_scale',
            torch.tensor(Config.BNN_DELTA_SCALE, dtype=torch.float32)
        )

    # ── core forward ──────────────────────────────────────────────────────────
    def forward(self, x):
        """x: (..., 6) → (..., 4).  Dropout active when self.training=True."""
        x = x / self.input_scale.to(x.device)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x) * self.delta_scale.to(x.device)

    # ── core batched MC inference ─────────────────────────────────────────────
    @torch.no_grad()
    def _mc_batch(self, xu_batch: torch.Tensor, num_samples=None):
        """
        xu_batch : (B, 6) on DEVICE  (B can be 1 for single-step or H for horizon)
        Returns  : mean (B, 4), var_tr (B,)  — numpy, on CPU

        Single (M×B, 6) forward replaces M serial forward calls.
        Dropout produces different masks for each of the M copies because
        nn.Dropout samples independently per element.
        """
        self.train()          # enable dropout — set once, not per-call
        M = int(self._M if num_samples is None else num_samples)
        if M < 2:
            raise ValueError("MC-dropout covariance requires at least two samples")
        B = xu_batch.shape[0]

        # (M, B, 6) → (M*B, 6): no data copy, just view + contiguous
        xu_flat = xu_batch.unsqueeze(0).expand(M, B, 6).reshape(M * B, 6)
        preds   = self.forward(xu_flat).view(M, B, 4)   # (M, B, 4)

        mean   = preds.mean(0)                            # (B, 4)
        # Dimensionless epistemic dispersion.  State increments have different
        # physical units (m, m, m/s, rad); summing their raw variances would be
        # unit-inconsistent.  ``delta_scale`` is the fixed output scale already
        # used by the predictor, so D^{-1} Δs is dimensionless.
        normalized = preds / self.delta_scale.to(preds.device)
        var_tr = normalized.var(0, unbiased=True).sum(1)  # (B,)

        return mean.cpu().numpy(), var_tr.cpu().numpy()

    # ── single-step: for data collection, heatmap, robustness eval ───────────
    def predict_uncertainty(self, s_np, a_np, num_samples=None):
        """(s, a) → (mean_Δs: R^4, var_trace: float)"""
        xu = torch.tensor(
            np.concatenate([s_np, a_np]),
            dtype=torch.float32, device=Config.DEVICE
        ).unsqueeze(0)                      # (1, 6)
        mean, var_tr = self._mc_batch(xu, num_samples=num_samples)
        return mean[0], float(var_tr[0])

    # ── horizon-batch: for FedUMPC _cost() — THE fast path ───────────────────
    def predict_horizon(self, states: np.ndarray, actions: np.ndarray,
                        num_samples=None):
        """
        states  : (H, 4)  predicted states along candidate trajectory
        actions : (H, 2)  candidate control sequence

        Returns
        -------
        means   : (H, 4)  np — predicted increments μ_i(k+h)
        var_trs : (H,)    np — uncertainty scalars   U_i(k+h)

        One call replaces H × M serial forward() calls.
        Used exclusively inside FedUMPCController._cost().
        """
        xu = torch.tensor(
            np.concatenate([states, actions], axis=1),   # (H, 6)
            dtype=torch.float32, device=Config.DEVICE
        )
        return self._mc_batch(xu, num_samples=num_samples)

    # ── full covariance (logging / paper Eq.10, not in hot path) ─────────────
    @torch.no_grad()
    def predict_with_uncertainty(self, s_np, a_np, num_samples=None):
        """Return physical covariance and the dimensionless normalized trace."""
        xu = torch.tensor(
            np.concatenate([s_np, a_np]),
            dtype=torch.float32, device=Config.DEVICE
        ).unsqueeze(0)
        self.train()
        M = int(self._M if num_samples is None else num_samples)
        if M < 2:
            raise ValueError("MC-dropout covariance requires at least two samples")
        xu_exp  = xu.expand(M, 1, 6).reshape(M, 6)
        preds   = self.forward(xu_exp)        # (M, 4)
        mean    = preds.mean(0)
        diff    = preds - mean.unsqueeze(0)
        cov     = (diff.T @ diff) / (M - 1)
        scale = self.delta_scale.to(diff.device)
        normalized_trace = (diff / scale).var(0, unbiased=True).sum()
        return (mean.cpu().numpy(), cov.cpu().numpy(),
                normalized_trace.item())


class EnsemblePredictor:
    """Deterministic deep-ensemble predictor with the BNN inference interface.

    This controlled baseline isolates uncertainty-source choice: independently
    initialized dynamics networks replace MC dropout, while the downstream MPC
    candidate set, objective weights, and margin map remain unchanged.
    """

    def __init__(self, members):
        if len(members) < 2:
            raise ValueError("an ensemble requires at least two members")
        self.members = list(members)
        self.delta_scale = self.members[0].delta_scale

    @torch.no_grad()
    def _predict(self, states, actions):
        xu = torch.as_tensor(
            np.concatenate([states, actions], axis=1),
            dtype=torch.float32, device=Config.DEVICE)
        outputs = []
        for member in self.members:
            member.eval()
            outputs.append(member(xu))
        stack = torch.stack(outputs, dim=0)
        mean = stack.mean(dim=0)
        scale = self.members[0].delta_scale.to(stack.device)
        var_trace = (stack / scale).var(dim=0, unbiased=True).sum(dim=1)
        return mean.cpu().numpy(), var_trace.cpu().numpy()

    def predict_horizon(self, states, actions, num_samples=None):
        del num_samples
        return self._predict(np.asarray(states), np.asarray(actions))

    def predict_uncertainty(self, state, action, num_samples=None):
        del num_samples
        mean, var_trace = self._predict(
            np.asarray(state, dtype=np.float32)[None, :],
            np.asarray(action, dtype=np.float32)[None, :])
        return mean[0], float(var_trace[0])
