"""Flow Engine — Component 5 of the Geometric Causal Architecture.

Responsibility:
  Navigate the manifold in response to a query to produce meaning as a
  continuous trajectory.  This IS the reasoning process.

What it is NOT responsible for:
  Storing anything.
  Creating language.
  Modifying the manifold (read-only access to C2).

The flow process:
  1. LOCATE    — inject the query into M via kNN resonance; P₀ = nearest point
  2. ORIENT    — V₀ = causal_direction(P₀, attractor) or gravity-based default
  3. INTEGRATE — run Euler-Maruyama: dP = μ(P,t)dt + σ(P,t)dW
  4. TERMINATE — stop when convergence/budget/revisit reached
  5. RETURN    — Trajectory handed to Resonance Layer (C6)

Termination conditions (first satisfied wins):
  velocity_threshold  : ‖V‖ < threshold for `patience` consecutive steps
  revisit_detected    : trajectory revisits a previous region
  max_steps           : hard ceiling on integration steps
  attractor_reached   : position enters the attractor basin
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

from .forces import ForceComputer
from .query import DIM, FlowStep, Query, Trajectory
from .sde import SDESolver


# ─────────────────────────────────────────────────────────────────────────────
# FlowEngine
# ─────────────────────────────────────────────────────────────────────────────

class FlowEngine:
    """Navigate M(t) via SDE to produce meaning as a trajectory.

    Parameters
    ----------
    manifold : LivingManifold
        The living manifold to navigate (read-only).
    max_steps : int
        Hard step limit before forced termination.  Default 200.
    dt : float
        Euler-Maruyama time step.  Default 0.05.
    velocity_threshold : float
        Speed below which flow is considered converged.  Default 0.001.
    patience : int
        Consecutive slow steps required to trigger velocity termination.
        Default 5.
    momentum : float
        γ for contextual momentum force.  Default 0.85.
    diffusion_scale : float
        Maximum diffusion (at zero density).  Default 0.05.
    revisit_radius : float
        Radius within which a position is considered a revisit.  Default 0.3.
    attractor_radius : float
        Radius within which we consider the attractor "reached".  Default 0.5.
    force_weights : tuple of 4 floats or None
        (w_gravity, w_causal, w_momentum, w_repulsion).  Default None = (0.4, 0.2, 0.3, 0.1).
    seed : int or None
        Random seed for the SDE solver.  Default None.
    """

    def __init__(
        self,
        manifold,
        max_steps: int = 200,
        dt: float = 0.05,
        velocity_threshold: float = 0.001,
        patience: int = 5,
        momentum: float = 0.85,
        diffusion_scale: float = 0.05,
        revisit_radius: float = 0.3,
        attractor_radius: float = 0.5,
        min_attractor_dist: float = 0.5,
        force_weights: Optional[Tuple[float, float, float, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._manifold = manifold
        self.max_steps = max_steps
        self.dt = dt
        self.velocity_threshold = velocity_threshold
        self.patience = patience
        self.momentum = momentum
        self.diffusion_scale = diffusion_scale
        self.revisit_radius = revisit_radius
        self.attractor_radius = attractor_radius
        self.min_attractor_dist = min_attractor_dist
        self._force_weights = force_weights

        rng = np.random.default_rng(seed)
        self._forces = ForceComputer()
        self._sde = SDESolver(
            dt=dt,
            diffusion_scale=diffusion_scale,
            rng=rng,
        )

    # ── Public interface ─────────────────────────────────────────────── #

    def flow(self, query: Query) -> Trajectory:
        """Run the flow from *query* and return the reasoning trajectory.

        Parameters
        ----------
        query : Query
            The query to reason about.

        Returns
        -------
        Trajectory
            Ordered sequence of FlowSteps representing meaning as a path.
        """
        # ── Step 1: Locate query on manifold ────────────────────────── #
        p0 = self._locate(query.vector)

        # ── Step 2: Find initial velocity ───────────────────────────── #
        v0 = self._initial_velocity(p0, query)

        # ── Step 3: Find attractor position ────────────────────────── #
        attractor = self._find_attractor(p0, query)

        # ── Step 4: Integrate SDE ───────────────────────────────────── #
        steps, reason = self._integrate(p0, v0, attractor)

        return Trajectory(
            steps=steps,
            query=query,
            termination_reason=reason,
        )

    # ── Internal helpers ─────────────────────────────────────────────── #

    def _locate(self, vector: np.ndarray) -> np.ndarray:
        """Find the manifold anchor closest to *vector*."""
        nearest = self._manifold.nearest(vector, k=1)
        if nearest:
            _label, anchor = nearest[0]
            return anchor.copy()
        return vector.copy()

    def _initial_velocity(self, p0: np.ndarray, query: Query) -> np.ndarray:
        """Compute V₀ pointing from P₀ toward the response attractor."""
        attractor_pos = self._find_attractor(p0, query)
        direction = attractor_pos - p0
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            # Fall back: use semantic gravity direction
            gravity = self._forces.semantic_gravity(p0, self._manifold)
            norm_g = float(np.linalg.norm(gravity))
            if norm_g > 1e-12:
                return gravity * self.dt
            # Last resort: small random perturbation
            return np.random.default_rng().standard_normal(DIM) * self.dt * 0.01
        return (direction / norm) * self.dt

    def _find_attractor(self, p0: np.ndarray, query: Query) -> np.ndarray:
        """Identify the response-attractor position in M.

        Strategy:
          1. If query.attractor_label is set, use its position.
          2. Otherwise, find the densest point among the k nearest neighbours
             of P₀ that is at least min_attractor_dist away from P₀
             (dense = crystallised = reliable anchor for an answer).
          3. If no such point exists, fall back to the farthest dense point
             among the k nearest.
        """
        if query.attractor_label is not None:
            try:
                return self._manifold.position(query.attractor_label).copy()
            except (KeyError, AttributeError):
                pass  # fall through to automatic selection

        # Automatic: pick densest of neighbours that is far enough from P₀
        neighbors = self._manifold.nearest(p0, k=20)
        if not neighbors:
            return p0.copy()

        best_vec = None
        best_density = -1.0

        for _, vec in neighbors:
            dist = float(np.linalg.norm(vec - p0))
            if dist < self.min_attractor_dist:
                continue  # skip the query point itself and immediate surroundings
            d = self._manifold.density(vec)
            if d > best_density:
                best_density = d
                best_vec = vec

        if best_vec is not None:
            return best_vec.copy()

        # Fallback: pick the farthest point among neighbors
        max_dist = -1.0
        fallback_vec = neighbors[0][1]
        for _, vec in neighbors:
            dist = float(np.linalg.norm(vec - p0))
            if dist > max_dist:
                max_dist = dist
                fallback_vec = vec
        return fallback_vec.copy()

    def _integrate(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        attractor: np.ndarray,
    ) -> Tuple[List[FlowStep], str]:
        """Run Euler-Maruyama loop and collect FlowSteps."""
        steps: List[FlowStep] = []
        position = p0.copy()
        velocity = v0.copy()
        slow_count = 0
        visited: List[np.ndarray] = []  # previous positions for revisit detection
        t = 0.0

        for _i in range(self.max_steps):
            kappa = self._manifold.curvature(position)
            steps.append(
                FlowStep(
                    position=position.copy(),
                    velocity=velocity.copy(),
                    time=t,
                    curvature=kappa,
                )
            )

            # ── Termination: attractor reached ──────────────────────── #
            dist_to_attractor = float(np.linalg.norm(position - attractor))
            if dist_to_attractor < self.attractor_radius:
                return steps, "attractor_reached"

            # ── Termination: revisit detected ───────────────────────── #
            if self._is_revisit(position, visited):
                return steps, "revisit_detected"

            # ── Compute drift ────────────────────────────────────────── #
            drift = self._forces.combined_drift(
                position,
                velocity,
                self._manifold,
                weights=self._force_weights,
            )

            # ── SDE step ─────────────────────────────────────────────── #
            new_position, new_velocity = self._sde.step(position, drift, self._manifold)

            # ── Termination: velocity threshold ──────────────────────── #
            speed = float(np.linalg.norm(new_velocity))
            if speed < self.velocity_threshold:
                slow_count += 1
                if slow_count >= self.patience:
                    # Record final step then terminate
                    t += self.dt
                    steps.append(
                        FlowStep(
                            position=new_position.copy(),
                            velocity=new_velocity.copy(),
                            time=t,
                            curvature=self._manifold.curvature(new_position),
                        )
                    )
                    return steps, "velocity_threshold"
            else:
                slow_count = 0

            visited.append(position.copy())
            position = new_position
            velocity = new_velocity
            t += self.dt

        return steps, "max_steps"

    @staticmethod
    def _is_revisit(
        position: np.ndarray,
        visited: List[np.ndarray],
        min_steps_back: int = 10,
    ) -> bool:
        """True if *position* is close to a previously visited location.

        Only checks positions that are at least *min_steps_back* steps old
        to avoid triggering on neighbouring steps.
        """
        if len(visited) < min_steps_back:
            return False
        for prev in visited[:-min_steps_back]:
            if float(np.linalg.norm(position - prev)) < 0.3:
                return True
        return False
