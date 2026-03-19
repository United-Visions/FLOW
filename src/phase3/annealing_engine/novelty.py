"""Novelty estimator for the Annealing Engine.

Novelty measures how surprising a new experience is given the current
manifold geometry.  It drives the magnitude of the deformation applied:

  high novelty → large deformation   (manifold must accommodate new shape)
  low  novelty → small deformation   (manifold already covers this territory)

Two signals are combined:

  1. Distance-based novelty
     distNovelty(E) = 1 - exp(-d_min / σ_d)
     where d_min = distance to nearest neighbor, σ_d = mean inter-point
     distance in the local neighborhood.

  2. Density-based novelty
     densNovelty(E) = 1 - ρ(P)
     where ρ(P) is the local density at the located position P.

Combined:
  novelty(E) = w_d · distNovelty + w_d_complement · densNovelty
             = 0.6 · distNovelty + 0.4 · densNovelty

Both components live in [0, 1]; the weighted sum is also in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class NoveltyResult:
    """Outcome of a novelty estimation.

    Attributes
    ----------
    score           : combined novelty in [0, 1]
    distance_score  : distance-based component in [0, 1]
    density_score   : density-based component in [0, 1]
    nearest_dist    : distance to the closest existing point
    local_density   : density value at the query position
    """

    score: float
    distance_score: float
    density_score: float
    nearest_dist: float
    local_density: float


class NoveltyEstimator:
    """Estimate the novelty of a new experience vector relative to the manifold.

    Parameters
    ----------
    weight_distance : float
        Weight for the distance-based novelty component.  Default 0.6.
    weight_density : float
        Weight for the density-based novelty component.  Default 0.4.
    sigma_scale : float
        Scaling factor for the exponential distance decay.  Higher values
        make the score less sensitive to small distances.  Default 1.0.
    """

    def __init__(
        self,
        weight_distance: float = 0.6,
        weight_density: float = 0.4,
        sigma_scale: float = 1.0,
    ) -> None:
        if abs(weight_distance + weight_density - 1.0) > 1e-9:
            raise ValueError("weight_distance + weight_density must equal 1.0")
        if sigma_scale <= 0:
            raise ValueError(f"sigma_scale must be positive, got {sigma_scale}")

        self.weight_distance = weight_distance
        self.weight_density = weight_density
        self.sigma_scale = sigma_scale

    # ------------------------------------------------------------------ #
    # Core interface                                                       #
    # ------------------------------------------------------------------ #

    def estimate(
        self,
        position: np.ndarray,
        neighbor_positions: List[np.ndarray],
        local_density: float,
    ) -> NoveltyResult:
        """Compute the novelty score for *position* given local context.

        Parameters
        ----------
        position          : 104D query vector
        neighbor_positions: list of nearby point vectors (may be empty)
        local_density     : ρ at the query position (from manifold.density)

        Returns
        -------
        NoveltyResult with score in [0, 1].
        """
        # Distance-based component
        if not neighbor_positions:
            dist_score = 1.0
            nearest_dist = float("inf")
        else:
            dists = [
                float(np.linalg.norm(position - nb))
                for nb in neighbor_positions
            ]
            nearest_dist = min(dists)
            # Use a fixed characteristic scale so that absolute distance
            # determines novelty (far = novel, close = familiar).
            # sigma_scale == 1.0 means 1 unit of distance → ~63 % novelty.
            sigma = max(self.sigma_scale, 1e-9)
            dist_score = 1.0 - math.exp(-nearest_dist / sigma)

        # Density-based component
        density_score = float(np.clip(1.0 - local_density, 0.0, 1.0))

        # Combine
        score = float(
            self.weight_distance * dist_score
            + self.weight_density * density_score
        )
        score = float(np.clip(score, 0.0, 1.0))

        return NoveltyResult(
            score=score,
            distance_score=dist_score,
            density_score=density_score,
            nearest_dist=nearest_dist,
            local_density=float(local_density),
        )

    # ------------------------------------------------------------------ #
    # Gradient toward consistency                                          #
    # ------------------------------------------------------------------ #

    def consistency_gradient(
        self,
        position: np.ndarray,
        neighbor_positions: List[np.ndarray],
        neighbor_weights: List[float] | None = None,
    ) -> np.ndarray:
        """Compute the deformation direction that improves geometric consistency.

        The gradient pulls the experience toward the weighted centroid of its
        nearest neighbours, encoding the intuition that similar experiences
        should cluster together.

        For an isolated point (no neighbors), returns a zero vector.

        Parameters
        ----------
        position          : 104D query vector (the experience position)
        neighbor_positions: nearby point vectors
        neighbor_weights  : optional weight per neighbor (default: uniform)

        Returns
        -------
        104D unit-normalised direction vector (zeros if no neighbours).
        """
        if not neighbor_positions:
            return np.zeros_like(position)

        n = len(neighbor_positions)
        if neighbor_weights is None:
            weights = np.ones(n, dtype=float)
        else:
            weights = np.array(neighbor_weights, dtype=float)

        # Guard against all-zero weights
        w_sum = float(weights.sum())
        if w_sum < 1e-12:
            weights = np.ones(n, dtype=float)
            w_sum = float(n)

        weights = weights / w_sum
        centroid = sum(w * nb for w, nb in zip(weights, neighbor_positions))
        direction = centroid - position

        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return np.zeros_like(position)
        return direction / norm

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NoveltyEstimator("
            f"w_dist={self.weight_distance}, "
            f"w_dens={self.weight_density}, "
            f"σ_scale={self.sigma_scale})"
        )
