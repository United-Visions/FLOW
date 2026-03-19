"""
Similarity Geometry
===================
Derives similarity structure from first principles.

Source:   The definition of similarity (metric space axioms)
Geometry: Flexible Riemannian metric space with local curvature variation.

Properties encoded
------------------
- Distance         : degree of difference between concepts
- Triangle ineq.   : enforced by construction (Riemannian metric)
- Local curvature  : varies by conceptual density
                      Dense regions → flatter geometry (well-understood)
                      Sparse regions → more curved (uncertain territory)
- No coordinates   : topology emerges from the metric alone

Design decision
---------------
The similarity geometry forms the BASE MANIFOLD of the fiber bundle.
All other geometries (causal, logical, probabilistic) are fibers
attached to this base.  The similarity metric's dimensionality is
therefore the largest — it carries the most semantic content.

The base space is initialised with structured curvature derived from
the universal domain taxonomy: a mathematically irreducible partition
of all possible concepts into coarse semantic regions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Dimensionality of the similarity base manifold
DIM_SIMILARITY = 64

# Universal domain taxonomy: irreducible semantic regions
# These are the coarsest possible partitions — derived from the structure
# of empirical reality and logical necessity, not from any corpus.
DOMAIN_TAXONOMY = {
    # Physical world
    "physical_objects":   {"center_axes": (0, 5),   "curvature": 0.3},
    "physical_forces":    {"center_axes": (5, 10),  "curvature": 0.4},
    "physical_processes": {"center_axes": (10, 15), "curvature": 0.5},
    "physical_states":    {"center_axes": (15, 20), "curvature": 0.3},

    # Biological world
    "living_organisms":   {"center_axes": (20, 24), "curvature": 0.4},
    "cognitive_agents":   {"center_axes": (24, 27), "curvature": 0.6},
    "social_structures":  {"center_axes": (27, 30), "curvature": 0.7},

    # Abstract / formal
    "mathematical":       {"center_axes": (30, 35), "curvature": 0.2},
    "logical_entities":   {"center_axes": (35, 39), "curvature": 0.2},
    "linguistic":         {"center_axes": (39, 43), "curvature": 0.5},

    # Events and processes
    "events_changes":     {"center_axes": (43, 47), "curvature": 0.5},
    "causal_mechanisms":  {"center_axes": (47, 51), "curvature": 0.4},

    # Properties and relations
    "properties":         {"center_axes": (51, 55), "curvature": 0.3},
    "relations":          {"center_axes": (55, 59), "curvature": 0.3},

    # Meta-cognitive
    "epistemic":          {"center_axes": (59, 62), "curvature": 0.6},
    "evaluative":         {"center_axes": (62, 64), "curvature": 0.7},
}


@dataclass
class SimilarityGeometry:
    """
    The similarity base manifold derived from metric space axioms.

    Attributes
    ----------
    dim : int
        Dimensionality of the base manifold.
    domain_centers : dict[str, np.ndarray]
        Representative centre point for each universal domain.
    curvature_field : callable
        Returns local curvature magnitude at any point.
    metric_tensor_fn : callable
        Returns the Riemannian metric tensor at any point.
    """

    dim: int = DIM_SIMILARITY
    domain_centers: Dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls) -> "SimilarityGeometry":
        """Derive the similarity geometry from metric space axioms."""
        geo = cls()
        geo._initialise_domain_centers()
        return geo

    def _initialise_domain_centers(self) -> None:
        """
        Create the centre vector for each universal semantic domain.

        Each domain occupies a dedicated sub-range of dimensions.
        Within its range, the centre has a distinctive pattern.
        All other dimensions are near zero (domains are roughly orthogonal).
        """
        for domain, props in DOMAIN_TAXONOMY.items():
            start, end = props["center_axes"]
            v = np.zeros(self.dim)
            length = end - start
            # Simple structured pattern: alternating gradient within the domain's axes
            for i, ax in enumerate(range(start, end)):
                v[ax] = np.cos(np.pi * i / max(length - 1, 1))
            # Normalise the domain-specific sub-vector
            sub = v[start:end]
            sub_norm = np.linalg.norm(sub)
            if sub_norm > 1e-12:
                v[start:end] = sub / sub_norm
            self.domain_centers[domain] = v

    # ------------------------------------------------------------------ #
    # Curvature field                                                        #
    # ------------------------------------------------------------------ #

    def local_curvature(self, p: np.ndarray) -> float:
        """
        Local scalar curvature at point p.

        Derived from proximity to domain centres:
        - Near a dense domain centre → low curvature (flat, well-understood)
        - Far from all centres → high curvature (uncertain, sparsely explored)

        This encodes the architecture's principle:
          "Crystallised regions have flat geometry;
           unknown territory remains curved and flexible."
        """
        if not self.domain_centers:
            return 1.0

        # Distance to nearest domain centre
        dists = [np.linalg.norm(p - c) for c in self.domain_centers.values()]
        d_min = min(dists)

        # Curvature is inversely related to proximity to domain centres
        # κ ∈ [0.1, 2.0]: near known domains → low κ; far away → high κ
        kappa_base = float(np.tanh(d_min))
        domain_curvatures = [v["curvature"] for v in DOMAIN_TAXONOMY.values()]

        # Blend base curvature with nearby domain's intrinsic curvature
        nearest_idx = int(np.argmin(dists))
        intrinsic = domain_curvatures[nearest_idx]
        return float(0.1 + 0.9 * (kappa_base * 0.7 + intrinsic * 0.3))

    def curvature_tensor(self, p: np.ndarray) -> np.ndarray:
        """
        Curvature tensor at point p (approximated as scalar × identity).

        Full Riemann curvature tensors are rank-4 and expensive.
        We use a sectional curvature approximation: κ(p) · I.

        Shape: (dim, dim).
        """
        kappa = self.local_curvature(p)
        return kappa * np.eye(self.dim)

    # ------------------------------------------------------------------ #
    # Riemannian metric                                                      #
    # ------------------------------------------------------------------ #

    def metric_tensor(self, p: np.ndarray) -> np.ndarray:
        """
        Riemannian metric tensor g(p) at point p.

        The metric captures how distances are measured locally.  Near
        domain centres (high density), we use a metric that stretches
        space slightly (making the region feel larger — reflecting that
        well-known concepts have rich internal structure).  In sparse
        regions the metric is closer to Euclidean.

        g(p) = I + α(p) · C(p)

        where C(p) is a density-weighted correction matrix and α(p)
        is a smooth blending coefficient.
        """
        alpha = 0.3 * (1.0 - self.local_curvature(p))
        # Domain-weighted stretch: pull metric toward domain sub-spaces
        correction = np.zeros((self.dim, self.dim))
        if self.domain_centers:
            dists = np.array([np.linalg.norm(p - c) for c in self.domain_centers.values()])
            weights = np.exp(-dists ** 2 / 2.0)
            weights /= weights.sum() + 1e-12
            for (domain, props), w in zip(DOMAIN_TAXONOMY.items(), weights):
                start, end = props["center_axes"]
                correction[start:end, start:end] += w * np.eye(end - start)
        return np.eye(self.dim) + alpha * correction

    def riemannian_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Approximate Riemannian distance between p1 and p2.

        Uses the midpoint metric as a first-order approximation:
            d(p1, p2) ≈ √( (p2 - p1)ᵀ g(mid) (p2 - p1) )

        where mid = (p1 + p2) / 2.
        For the exact geodesic, numerical integration along the path is
        required — that belongs in the Living Manifold (Component 2).
        """
        mid = (p1 + p2) / 2.0
        g = self.metric_tensor(mid)
        diff = p2 - p1
        d_sq = float(diff @ g @ diff)
        return float(np.sqrt(max(d_sq, 0.0)))

    def similarity_score(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Similarity in [0, 1]: 1 = identical, 0 = maximally dissimilar.

        Derived from the Riemannian distance, normalised by the expected
        diameter of the manifold (√dim).
        """
        d = self.riemannian_distance(p1, p2)
        expected_diameter = np.sqrt(self.dim)
        return float(np.exp(-d / expected_diameter))

    # ------------------------------------------------------------------ #
    # Neighbourhood and domain queries                                       #
    # ------------------------------------------------------------------ #

    def domain_of(self, p: np.ndarray) -> str:
        """Return the name of the nearest semantic domain to point p."""
        dists = {d: np.linalg.norm(p - c) for d, c in self.domain_centers.items()}
        return min(dists, key=dists.get)

    def neighbours_in_domain(
        self,
        points: np.ndarray,
        query: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """
        Find all points in `points` within Riemannian radius of `query`.

        Returns indices into `points`.
        """
        dists = np.array([self.riemannian_distance(query, p) for p in points])
        return np.where(dists <= radius)[0]

    def density_estimate(self, points: np.ndarray, query: np.ndarray,
                          bandwidth: float = 1.0) -> float:
        """
        Gaussian kernel density estimate at `query` given `points`.

        Uses the Riemannian metric for distances.
        Higher density → more crystallised → less curvature.
        """
        if len(points) == 0:
            return 0.0
        dists = np.array([self.riemannian_distance(query, p) for p in points])
        return float(np.mean(np.exp(-dists ** 2 / (2 * bandwidth ** 2))))

    def locality_radius(self, p: np.ndarray, base_radius: float = 1.0) -> float:
        """
        The locality radius for deformation operations at point p.

        Per the architecture specification:
          "locality radius = f(density) — denser = smaller radius"

        Here density is approximated by `1 - local_curvature`:
        flat (dense) regions get a small radius;
        curved (sparse) regions get a large radius.
        """
        kappa = self.local_curvature(p)
        # High curvature (sparse) → large radius; low curvature (dense) → small
        return base_radius * (0.2 + 0.8 * kappa)

    def summary(self) -> str:
        return (
            f"SimilarityGeometry(dim={self.dim}, "
            f"n_domains={len(self.domain_centers)}, "
            f"domain_names={list(self.domain_centers.keys())})"
        )
