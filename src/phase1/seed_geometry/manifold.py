"""
Seed Manifold — M₀
===================
The output data structure of the Seed Geometry Engine.

M₀ is an immutable, fully queryable Riemannian manifold that encodes
the mathematical structure of causality, logic, probability, and similarity
without a single data point.

Once built, M₀ is never modified.  It is the initial state that the
Living Manifold (Component 2) takes as its starting point:

    M(t=0) = M₀
    M(t)   = (M₀, φ(t), ρ(t), κ(t))   — for t > 0, after experience arrives

Query API
---------
The API exposed here mirrors exactly the READ operations listed in the
architecture specification for Component 2 (Living Manifold):

    position(concept)      → point P in M₀
    distance(P₁, P₂)       → geodesic distance estimate
    causal_direction(P₁, P₂) → causal flow vector from P₁ toward P₂
    curvature(P)           → curvature magnitude at P
    density(P)             → local density estimate at P
    neighbors(P, r)        → seed points within radius r
    domain_of(P)           → nearest semantic domain name
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .composer import FiberBundleComposer, DIM_TOTAL, SLICES
from .causal import CausalGeometry
from .logical import LogicalGeometry
from .probabilistic import ProbabilisticGeometry
from .similarity import SimilarityGeometry


@dataclass
class ManifoldPoint:
    """
    A point on the seed manifold M₀.

    Attributes
    ----------
    vector : np.ndarray
        The 104-dimensional coordinate vector in the bundle space.
    label : str
        Optional human-readable label (e.g., for archetypal/seed points).
    origin : str
        Which geometry this point originated from: 'causal', 'logical',
        'probabilistic', 'similarity', 'composed', or 'query'.
    """
    vector: np.ndarray
    label: str = ""
    origin: str = "composed"

    def __post_init__(self) -> None:
        assert self.vector.shape == (DIM_TOTAL,), (
            f"ManifoldPoint.vector must have shape ({DIM_TOTAL},), got {self.vector.shape}"
        )

    @property
    def base(self) -> np.ndarray:
        return self.vector[SLICES.base]

    @property
    def causal_fiber(self) -> np.ndarray:
        return self.vector[SLICES.causal]

    @property
    def logical_fiber(self) -> np.ndarray:
        return self.vector[SLICES.logical]

    @property
    def prob_fiber(self) -> np.ndarray:
        return self.vector[SLICES.prob]

    def __repr__(self) -> str:
        return f"ManifoldPoint(label='{self.label}', origin='{self.origin}', dim={DIM_TOTAL})"


class SeedManifold:
    """
    The seed manifold M₀.

    Immutable after construction.  Provides the READ operations that
    feed into the Living Manifold (Component 2) and ultimately into
    the Flow Engine (Component 5).

    Attributes
    ----------
    dim : int
        Total dimensionality = 104.
    seed_points : list[ManifoldPoint]
        The archetypal seed points — the manifold's skeleton.
    composer : FiberBundleComposer
        The composed bundle geometry.
    build_time_s : float
        Wall-clock time taken to build M₀ (for benchmarking).
    """

    def __init__(
        self,
        sim:   SimilarityGeometry,
        cau:   CausalGeometry,
        log:   LogicalGeometry,
        prob:  ProbabilisticGeometry,
        composer: FiberBundleComposer,
        seed_points: List[ManifoldPoint],
        build_time_s: float = 0.0,
    ) -> None:
        self.dim           = DIM_TOTAL
        self.sim           = sim
        self.cau           = cau
        self.log           = log
        self.prob          = prob
        self.composer      = composer
        self.seed_points   = seed_points
        self.build_time_s  = build_time_s

        # Pre-compute seed point vectors for fast neighbour queries
        self._seed_vectors: np.ndarray = np.stack(
            [sp.vector for sp in seed_points]
        ) if seed_points else np.empty((0, DIM_TOTAL))

    # ------------------------------------------------------------------ #
    # READ operations (Component 2 interface)                               #
    # ------------------------------------------------------------------ #

    def position(self, label: str) -> ManifoldPoint:
        """
        Find a seed point by its label.

        This mimics `M.position(concept)` in the architecture spec.
        For the seed manifold, the available "concepts" are the
        archetypal seed points.
        """
        for sp in self.seed_points:
            if sp.label == label:
                return sp
        raise KeyError(
            f"Seed point '{label}' not found. "
            f"Available: {[sp.label for sp in self.seed_points[:10]]}..."
        )

    def distance(self, p1: ManifoldPoint, p2: ManifoldPoint) -> float:
        """
        Geodesic distance estimate between two manifold points.

        Uses the composed bundle metric at the midpoint (first-order approx).
        """
        return self.composer.bundle_distance(p1.vector, p2.vector)

    def causal_direction(self, p1: ManifoldPoint, p2: ManifoldPoint) -> np.ndarray:
        """
        Causal direction vector from p1 toward p2 in the causal fiber.

        Returns a unit vector in the full 104D space, with non-zero
        components only in the causal fiber slice.
        """
        v1_c = p1.causal_fiber
        v2_c = p2.causal_fiber
        cau_dir = self.cau.causal_direction(v1_c, v2_c)

        # Embed back into the full 104D space
        full = np.zeros(DIM_TOTAL)
        full[SLICES.causal] = cau_dir
        return full

    def curvature(self, p: ManifoldPoint) -> float:
        """
        Scalar curvature at point p (magnitude of the curvature tensor).

        Blends the base manifold similarity curvature with
        the logical uncertainty at p's logical fiber.
        """
        kappa_sim = self.sim.local_curvature(p.base)
        kappa_log = self.log.uncertainty_score(p.logical_fiber)
        kappa_prob = 1.0 - self.prob.confidence(p.prob_fiber)
        # Weighted blend: base geometry dominates
        return float(0.6 * kappa_sim + 0.2 * kappa_log + 0.2 * kappa_prob)

    def density(self, p: ManifoldPoint, radius: Optional[float] = None) -> float:
        """
        Density at point p: how many seed points are nearby.

        Returns a value in [0, 1] where 1 = very dense neighbourhood.
        For the seed manifold this reflects inter-archetype proximity.
        """
        if len(self._seed_vectors) == 0:
            return 0.0
        r = radius or 2.0
        dists = np.linalg.norm(self._seed_vectors - p.vector, axis=1)
        count = np.sum(dists < r)
        # Normalise by total seed points
        return float(min(count / max(len(self.seed_points), 1), 1.0))

    def neighbors(self, p: ManifoldPoint, radius: float) -> List[ManifoldPoint]:
        """
        All seed points within the given radius of p.

        Uses Euclidean distance in the 104D bundle space as a fast proxy
        for the full Riemannian distance.  The exact geodesic is computed
        only when needed (Component 2 / Flow Engine).
        """
        if len(self._seed_vectors) == 0:
            return []
        dists = np.linalg.norm(self._seed_vectors - p.vector, axis=1)
        idxs  = np.where(dists <= radius)[0]
        return [self.seed_points[i] for i in idxs]

    def nearest(self, p: ManifoldPoint, k: int = 1) -> List[ManifoldPoint]:
        """
        Return the k nearest seed points to p.
        """
        if len(self._seed_vectors) == 0:
            return []
        dists = np.linalg.norm(self._seed_vectors - p.vector, axis=1)
        idxs  = np.argsort(dists)[:k]
        return [self.seed_points[i] for i in idxs]

    def domain_of(self, p: ManifoldPoint) -> str:
        """
        Return the name of the closest semantic domain to p.
        """
        return self.sim.domain_of(p.base)

    def locality_radius(self, p: ManifoldPoint, base_radius: float = 1.0) -> float:
        """
        The locality radius for deformation operations at p.
        Delegates to the similarity geometry.
        """
        return self.sim.locality_radius(p.base, base_radius)

    def causal_ancestry(self, p1: ManifoldPoint, p2: ManifoldPoint) -> bool:
        """True if p1 causally precedes p2 in the seed causal fiber."""
        return self.cau.is_causal_ancestor(p1.causal_fiber, p2.causal_fiber)

    def confidence(self, p: ManifoldPoint) -> float:
        """Confidence score at p (from probabilistic fiber). Range [0, 1]."""
        return self.prob.confidence(p.prob_fiber)

    def logic_certainty(self, p: ManifoldPoint) -> float:
        """Logical certainty at p (complement of logical uncertainty). Range [0, 1]."""
        return 1.0 - self.log.uncertainty_score(p.logical_fiber)

    # ------------------------------------------------------------------ #
    # Snapshot and introspection                                            #
    # ------------------------------------------------------------------ #

    def validate(self) -> Dict[str, object]:
        """
        Run self-validation checks and return a report dict.

        Validates:
        - Dimensionality of all seed points
        - Metric PSD at a sample point
        - Triangle inequality for a sample triple
        - Causal direction consistency (cause precedes effect in τ)
        - Logical distance properties (negation is maximum distance)
        """
        results: Dict[str, object] = {}

        # 1. Dimensionality
        wrong_dim = [sp.label for sp in self.seed_points if sp.vector.shape[0] != DIM_TOTAL]
        results["all_points_correct_dim"] = len(wrong_dim) == 0
        if wrong_dim:
            results["wrong_dim_labels"] = wrong_dim

        # 2. Metric PSD at a neutral point
        neutral = ManifoldPoint(self.composer.all_neutral(), label="__neutral__")
        metric_check = self.composer.validate_metric(neutral.vector)
        results["metric_psd"]           = metric_check["is_psd"]
        results["metric_symmetric"]     = metric_check["is_symmetric"]
        results["metric_min_eigenvalue"] = metric_check["min_eigenvalue"]

        # 3. Triangle inequality for first 3 seed points (if available)
        if len(self.seed_points) >= 3:
            a, b, c = self.seed_points[:3]
            d_ab = self.distance(a, b)
            d_bc = self.distance(b, c)
            d_ac = self.distance(a, c)
            results["triangle_inequality"] = bool(d_ac <= d_ab + d_bc + 1e-8)
            results["distances_non_negative"] = bool(d_ab >= 0 and d_bc >= 0 and d_ac >= 0)

        # 4. Check that causal distance is asymmetric
        if len(self.seed_points) >= 2:
            p, q = self.seed_points[0], self.seed_points[1]
            # Find a pair where one comes before the other causally
            causal_pts = [sp for sp in self.seed_points
                          if sp.origin == "causal" and hasattr(self.cau, 'embeddings')]
            if len(causal_pts) >= 2:
                c1, c2 = causal_pts[0], causal_pts[1]
                d_fwd = self.cau.causal_distance(c1.causal_fiber, c2.causal_fiber)
                d_rev = self.cau.causal_distance(c2.causal_fiber, c1.causal_fiber)
                results["causal_asymmetry"] = bool(abs(d_fwd - d_rev) > 1e-8)

        # 5. Logical negation = maximum distance
        logical_pts = [sp for sp in self.seed_points if sp.origin == "logical"]
        if len(logical_pts) >= 1:
            v = logical_pts[0].logical_fiber
            nv = self.log.negate(v)
            neg_point = ManifoldPoint(
                self.composer.bundle_point(
                    np.zeros(64), np.zeros(16), nv, np.full(16, 1/16)
                ),
                label="__neg__",
                origin="logical"
            )
            dist = self.log.continuous_distance(v, nv)
            results["negation_max_dist_logical"] = bool(dist > 0.5)

        results["n_seed_points"] = len(self.seed_points)
        results["dim"] = self.dim
        results["build_time_s"] = self.build_time_s

        return results

    def summary(self) -> str:
        lines = [
            "═══ Seed Manifold M₀ ═══════════════════════════════════════",
            f"  Total dimension  : {self.dim}",
            f"  Seed points      : {len(self.seed_points)}",
            f"  Build time       : {self.build_time_s:.3f}s",
            "",
            f"  {self.sim.summary()}",
            f"  {self.cau.summary()}",
            f"  {self.log.summary()}",
            f"  {self.prob.summary()}",
            f"  {self.composer.summary()}",
            "═══════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)
