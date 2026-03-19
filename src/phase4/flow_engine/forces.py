"""Force computation for the Flow Engine SDE drift vector μ.

The drift vector μ(P, t) is the sum of four geometrically-grounded forces:

    μ = F_gravity + F_causal + F_momentum + F_repulsion

Each force reads from the manifold (C2 READ operations only) and
returns a 104D vector contribution.

Force 1 — Semantic Gravity
    F_gravity(P) = Σᵢ  mᵢ · (Pᵢ − P) / max(‖Pᵢ − P‖², ε)
    mᵢ = density(Pᵢ)  (denser = more massive = stronger pull)
    Effect: nearby dense concept clusters attract the flow trajectory.

Force 2 — Causal Curvature
    F_causal(P) = κ(P) · causal_dir(P, nearest_cause)
    κ = scalar curvature at P
    Effect: the causal fiber bends the trajectory, preserving cause → effect order.

Force 3 — Contextual Momentum
    F_momentum(P) = γ · V_prev    γ ∈ (0, 1)
    Effect: meaning has inertia; themes persist across steps.

Force 4 — Contrast Repulsion
    F_repulsion(P) = −strength · Σⱼ contr(P, Pⱼ) · (Pⱼ − P)
    contr(P, Q) = logical contradiction score from the Boolean fiber (dims 80-87)
    Effect: contradictory regions repel the flow; logical coherence from geometry.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Manifold dimension slices (must match architect spec)
_SIM_SLICE = slice(0, 64)
_CAU_SLICE = slice(64, 80)
_LOG_SLICE = slice(80, 88)
_PRO_SLICE = slice(88, 104)

DIM = 104


# ─────────────────────────────────────────────────────────────────────────────
# ForceComputer
# ─────────────────────────────────────────────────────────────────────────────

class ForceComputer:
    """Compute the four drift forces for the SDE.

    Parameters
    ----------
    gravity_k : int
        Number of nearest neighbours used for semantic-gravity computation.
    repulsion_k : int
        Number of candidate contradictory points checked for contrast repulsion.
    repulsion_strength : float
        Scaling factor for the contrast-repulsion force.
    gravity_eps : float
        Minimum squared distance denominator (prevents singularity).
    causal_scale : float
        Multiplicative scale applied to the causal-curvature force.
    """

    def __init__(
        self,
        gravity_k: int = 8,
        repulsion_k: int = 5,
        repulsion_strength: float = 0.3,
        gravity_eps: float = 1e-4,
        causal_scale: float = 0.5,
    ) -> None:
        self.gravity_k = gravity_k
        self.repulsion_k = repulsion_k
        self.repulsion_strength = repulsion_strength
        self.gravity_eps = gravity_eps
        self.causal_scale = causal_scale

    # ── Force 1: Semantic Gravity ─────────────────────────────────────── #

    def semantic_gravity(
        self,
        position: np.ndarray,
        manifold,
    ) -> np.ndarray:
        """F_gravity = Σᵢ mᵢ · (Pᵢ − P) / max(‖Pᵢ − P‖², ε).

        Uses kNN lookup via manifold.nearest() then manifold.density() per point.
        """
        neighbors: List[Tuple[str, np.ndarray]] = manifold.nearest(
            position, k=self.gravity_k
        )
        force = np.zeros(DIM)
        for _label, vec in neighbors:
            diff = vec - position
            dist_sq = float(np.dot(diff, diff))
            mass = manifold.density(vec)
            force += mass * diff / max(dist_sq, self.gravity_eps)
        # Normalise so individual force magnitudes are comparable
        norm = float(np.linalg.norm(force))
        if norm > 1e-12:
            force /= norm
        return force

    # ── Force 2: Causal Curvature ─────────────────────────────────────── #

    def causal_curvature(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        manifold,
    ) -> np.ndarray:
        """F_causal = κ(P) · causal_direction(P, nearest_downstream).

        We pick the nearest point that is causally downstream of P (in the
        causal fiber sub-space dims 64-79).  If none found, we fall back
        to bending velocity toward the causal-fiber component of movement.
        """
        kappa: float = manifold.curvature(position)

        # Attempt to find a nearby causal descendant
        neighbors: List[Tuple[str, np.ndarray]] = manifold.nearest(
            position, k=self.gravity_k
        )
        causal_vec = np.zeros(DIM)
        for _label, vec in neighbors:
            direction = manifold.causal_direction(position, vec)
            # causal_direction returns 104D vector; non-zero means downstream
            if float(np.linalg.norm(direction)) > 1e-12:
                causal_vec = direction
                break

        # Scale by curvature
        force = self.causal_scale * kappa * causal_vec
        norm = float(np.linalg.norm(force))
        if norm > 1e-12:
            force /= norm
        return force

    # ── Force 3: Contextual Momentum ─────────────────────────────────── #

    def contextual_momentum(
        self,
        velocity: np.ndarray,
        gamma: float = 0.85,
    ) -> np.ndarray:
        """F_momentum = γ · V_prev — meaning has inertia."""
        return gamma * velocity

    # ── Force 4: Contrast Repulsion ───────────────────────────────────── #

    def contrast_repulsion(
        self,
        position: np.ndarray,
        manifold,
    ) -> np.ndarray:
        """F_repulsion = −strength · Σⱼ contr(P, Pⱼ) · (Pⱼ − P).

        Contradiction score between P and Q is measured in the logical fiber
        (dims 80-87).  Two points are contradictory when their Boolean-fiber
        projections are anti-correlated (cosine similarity ≈ −1).
        The force repels P away from maximally contradictory neighbours.
        """
        neighbors: List[Tuple[str, np.ndarray]] = manifold.nearest(
            position, k=self.repulsion_k
        )
        force = np.zeros(DIM)
        p_log = position[_LOG_SLICE]
        p_log_norm = float(np.linalg.norm(p_log))

        for _label, vec in neighbors:
            q_log = vec[_LOG_SLICE]
            q_log_norm = float(np.linalg.norm(q_log))
            if p_log_norm < 1e-12 or q_log_norm < 1e-12:
                continue
            # Cosine similarity in logical sub-space: −1 = contradiction
            cos_sim = float(np.dot(p_log, q_log) / (p_log_norm * q_log_norm))
            # Contradiction strength: 0 = identical, 1 = fully contradictory
            contr = max(0.0, -cos_sim)
            if contr < 1e-6:
                continue
            diff = vec - position
            # Repel: push P away from contradictory Q
            force -= self.repulsion_strength * contr * diff

        norm = float(np.linalg.norm(force))
        if norm > 1e-12:
            force /= norm
        return force

    # ── Combined drift μ ──────────────────────────────────────────────── #

    def combined_drift(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        manifold,
        weights: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """μ(P, t) = w₁F_gravity + w₂F_causal + w₃F_momentum + w₄F_repulsion.

        Parameters
        ----------
        weights : (w1, w2, w3, w4) or None
            Per-force weights.  Default (0.4, 0.2, 0.3, 0.1).
        """
        w1, w2, w3, w4 = weights if weights is not None else (0.4, 0.2, 0.3, 0.1)

        f_grav = self.semantic_gravity(position, manifold)
        f_cau  = self.causal_curvature(position, velocity, manifold)
        f_mom  = self.contextual_momentum(velocity)
        f_rep  = self.contrast_repulsion(position, manifold)

        drift = w1 * f_grav + w2 * f_cau + w3 * f_mom + w4 * f_rep
        return drift
