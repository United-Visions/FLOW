"""Query and Trajectory data types for the Flow Engine (C5).

A Query enters the system as a perturbation injected into the manifold:
  1. The query vector locates Q in M via resonance (kNN lookup)
  2. Initial velocity V₀ is set toward the response attractor region
  3. The SDE propagates from P₀ forward in continuous time

A FlowStep records one Euler-Maruyama integration step of the SDE:
  dP = μ(P, t) dt + σ(P, t) dW

A Trajectory is the complete ordered sequence of FlowSteps — meaning
as a continuous path through the manifold, handed to the Resonance Layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

DIM = 104  # must match manifold bundle dimension


# ─────────────────────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Query:
    """A query injected into the manifold as a perturbation.

    Attributes
    ----------
    vector          : 104D position vector — where Q fits in M
    label           : optional human-readable description
    attractor_label : optional label of the manifold region that anchors
                      the response; when None the engine chooses the nearest
                      dense region automatically
    """

    vector: np.ndarray
    label: str = ""
    attractor_label: Optional[str] = None

    def __post_init__(self) -> None:
        v = np.asarray(self.vector, dtype=float)
        if v.ndim != 1 or v.shape[0] != DIM:
            raise ValueError(
                f"Query vector must be 1D with {DIM} dimensions, got shape {v.shape}"
            )
        object.__setattr__(self, "vector", v)


# ─────────────────────────────────────────────────────────────────────────────
# FlowStep
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowStep:
    """One integration step of the Flow Engine SDE.

    Attributes
    ----------
    position   : 104D position on M at this step
    velocity   : 104D velocity vector (drift direction + magnitude)
    time       : continuous flow time t at this step (starts at 0)
    speed      : scalar ‖velocity‖ — derived from velocity on init
    curvature  : local manifold curvature at position (from C2)
    """

    position: np.ndarray
    velocity: np.ndarray
    time: float
    speed: float = field(init=False)
    curvature: float = 0.0

    def __post_init__(self) -> None:
        self.speed = float(np.linalg.norm(self.velocity))

    def __repr__(self) -> str:
        return (
            f"FlowStep(t={self.time:.3f}, speed={self.speed:.4f}, "
            f"curvature={self.curvature:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """The complete output of the Flow Engine — meaning as a path through M.

    Attributes
    ----------
    steps              : ordered FlowSteps P₀ … Pₙ
    query              : the query that initiated this flow
    total_time         : total elapsed flow time (= steps[-1].time)
    termination_reason : why the flow stopped; one of:
                         'velocity_threshold', 'revisit_detected',
                         'max_steps', 'attractor_reached', 'empty'
    """

    steps: List[FlowStep]
    query: Query
    total_time: float = 0.0
    termination_reason: str = "unknown"

    def __post_init__(self) -> None:
        if self.steps:
            self.total_time = self.steps[-1].time

    # ── Derived views ─────────────────────────────────────────────────── #

    @property
    def positions(self) -> List[np.ndarray]:
        """All position vectors in step order."""
        return [s.position for s in self.steps]

    @property
    def velocities(self) -> List[np.ndarray]:
        """All velocity vectors in step order."""
        return [s.velocity for s in self.steps]

    @property
    def as_position_time_pairs(self) -> List[Tuple[np.ndarray, float]]:
        """List of (position, time) pairs consumed by the Resonance Layer."""
        return [(s.position, s.time) for s in self.steps]

    @property
    def mean_speed(self) -> float:
        """Mean flow speed across all steps."""
        if not self.steps:
            return 0.0
        return float(np.mean([s.speed for s in self.steps]))

    @property
    def mean_curvature(self) -> float:
        """Mean local curvature encountered during the flow."""
        if not self.steps:
            return 0.0
        return float(np.mean([s.curvature for s in self.steps]))

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_steps={len(self.steps)}, "
            f"total_time={self.total_time:.3f}, "
            f"reason='{self.termination_reason}', "
            f"mean_speed={self.mean_speed:.4f})"
        )
