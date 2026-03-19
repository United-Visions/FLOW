"""Stochastic differential equation solver for the Flow Engine.

The flow is governed by an Euler-Maruyama discretisation of the SDE:

    dP = μ(P, t) dt + σ(P, t) dW

Where:
  μ(P, t) = combined drift from the ForceComputer
  σ(P, t) = diffusion tensor — scalar * I for numerical tractability
  dW       = Riemannian Brownian increment (Gaussian noise)

Diffusion magnitude:
  σ(P) = diffusion_scale · (1 − density(P))

  Sparse (unexplored) regions → high diffusion (creative, exploratory)
  Dense (crystallised) regions → low diffusion (focused, precise)

The solver is intentionally simple: one Euler step per call.  The Flow
Engine calls it in a loop, applying termination conditions between steps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

DIM = 104


class SDESolver:
    """Euler-Maruyama SDE step on the Riemannian manifold.

    Parameters
    ----------
    dt : float
        Time step size.  Default 0.05.
    diffusion_scale : float
        Maximum diffusion magnitude (reached in zero-density regions).
        Default 0.05.
    rng : np.random.Generator or None
        Random number generator.  If None, a default generator is created.
    """

    def __init__(
        self,
        dt: float = 0.05,
        diffusion_scale: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if diffusion_scale < 0:
            raise ValueError(f"diffusion_scale must be non-negative, got {diffusion_scale}")
        self.dt = dt
        self.diffusion_scale = diffusion_scale
        self._rng = rng if rng is not None else np.random.default_rng()

    # ── Core solver step ─────────────────────────────────────────────── #

    def step(
        self,
        position: np.ndarray,
        drift: np.ndarray,
        manifold,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply one Euler-Maruyama step.

        Parameters
        ----------
        position : 104D current position P
        drift    : 104D drift vector μ(P, t)  (from ForceComputer)
        manifold : LivingManifold (read-only)

        Returns
        -------
        (new_position, new_velocity)
          new_position   : P + μ·dt + σ·√dt·dW
          new_velocity   : the deterministic drift contribution μ·dt
                           (used as the velocity vector in FlowStep)
        """
        # Diffusion magnitude: high in sparse regions
        dens = manifold.density(position)
        sigma = self.diffusion_scale * (1.0 - dens)

        # Brownian increment: Gaussian noise scaled by √dt
        dW = self._rng.standard_normal(DIM)
        dW /= (float(np.linalg.norm(dW)) + 1e-12)  # unit direction

        # Deterministic displacement
        deterministic = drift * self.dt

        # Stochastic displacement
        stochastic = sigma * np.sqrt(self.dt) * dW

        new_position = position + deterministic + stochastic
        new_velocity = deterministic   # velocity = drift·dt for this step

        return new_position, new_velocity

    def diffusion_at(self, position: np.ndarray, manifold) -> float:
        """Return the diffusion magnitude σ(P) at *position*."""
        return float(self.diffusion_scale * (1.0 - manifold.density(position)))
