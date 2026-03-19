"""
Logical Geometry
================
Derives logical structure from first principles.

Source:   Propositional logic (Boolean algebra)
Geometry: n-dimensional Boolean hypercube embedded in continuous space.

Properties encoded
------------------
- Contradiction  : maximum distance (opposite vertices of the hypercube)
- Entailment     : directional proximity (A → B means B is reachable from A)
- Conjunction    : geometric intersection of two vertex sets
- Disjunction    : geometric union of two vertex sets
- Negation       : reflection across a hyperplane through the hypercube centre

Design decision
---------------
We use n = N_LOGICAL_DIMS bits.  2^n vertices represent the complete space
of logical state-combinations.  This is not about n specific propositions —
it is about the *shape* of logical space itself, which every proposition
lives within.  The vertices, edges, and faces of the hypercube ARE the
logical topology that the manifold must carry.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# Dimensionality of the logical fiber
# 2^N_LOGICAL_DIMS vertices in the hypercube
N_LOGICAL_DIMS = 8
DIM_LOGICAL = N_LOGICAL_DIMS   # the hypercube is embedded in its own n-space


@dataclass
class LogicalGeometry:
    """
    The logical geometric fiber derived from Boolean algebra.

    Attributes
    ----------
    n_dims : int
        Number of logical dimensions (hypercube side length = n_dims).
    dim : int
        Ambient dimensionality of this geometry (= n_dims for the hypercube).
    vertices : np.ndarray
        Shape (2**n_dims, n_dims).  Each row is a {0, 1}^n vertex.
    logical_metric : np.ndarray
        The metric tensor that enforces the hypercube's logical distances.
        Shape (n_dims, n_dims).
    """

    n_dims: int = N_LOGICAL_DIMS
    dim: int = DIM_LOGICAL
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    logical_metric: np.ndarray = field(default_factory=lambda: np.eye(DIM_LOGICAL))

    # ------------------------------------------------------------------ #
    # Construction                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls) -> "LogicalGeometry":
        """Derive the logical geometry from Boolean algebra."""
        geo = cls()
        geo._generate_hypercube_vertices()
        geo._compute_logical_metric()
        return geo

    def _generate_hypercube_vertices(self) -> None:
        """
        Generate all 2^n vertices of the n-dimensional Boolean hypercube.

        Vertex encoding: row i is the binary representation of integer i,
        padded to n_dims bits.  Each vertex corresponds to one complete
        truth-value assignment to n logical propositions.
        """
        n = 2 ** self.n_dims
        verts = np.zeros((n, self.n_dims), dtype=np.float64)
        for i in range(n):
            bits = [(i >> b) & 1 for b in range(self.n_dims)]
            verts[i] = bits
        self.vertices = verts

    def _compute_logical_metric(self) -> None:
        """
        Compute the logical metric tensor.

        The natural distance on the hypercube is the Hamming distance, which
        counts differing bit-positions.  In the continuous embedding this
        becomes the L1 metric scaled to match Hamming distances exactly.

        We use the identity metric here (Euclidean → L2 ≈ Hamming for {0,1}
        vectors) but normalised so that maximum distance (n_dims) maps to 1.
        """
        scale = 1.0 / self.n_dims
        self.logical_metric = scale * np.eye(self.dim)

    # ------------------------------------------------------------------ #
    # Core logical operations as geometry                                   #
    # ------------------------------------------------------------------ #

    def hamming_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Hamming distance between two logical points.

        Counts the number of logical dimensions that differ.
        Range: [0, n_dims].  Maximum = n_dims = maximum contradiction.
        """
        diff = np.abs(v1 - v2)
        # Allow continuous points by rounding to nearest vertex first
        return float(np.sum(diff > 0.5))

    def continuous_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Continuous logical distance between two points in logical space.

        Uses the logical metric tensor for smooth interpolation between
        the discrete Hamming vertices.  Normalised to [0, 1].
        """
        diff = v2 - v1
        return float(np.sqrt(diff @ self.logical_metric @ diff))

    def negate(self, v: np.ndarray) -> np.ndarray:
        """
        Logical NOT: reflection across the hypercube centre (0.5, ..., 0.5).

        ¬v = 1 - v  (for {0,1} vertices: flips all bits)
        For continuous points: reflection through the centroid.
        """
        return 1.0 - v

    def conjunction(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Logical AND: component-wise minimum (geometric intersection).

        For {0,1} vertices: v1 AND v2.
        For continuous points: element-wise minimum (conservative estimate).
        """
        return np.minimum(v1, v2)

    def disjunction(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Logical OR: component-wise maximum (geometric union).

        For {0,1} vertices: v1 OR v2.
        For continuous points: element-wise maximum.
        """
        return np.maximum(v1, v2)

    def entails(self, v1: np.ndarray, v2: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Check if v1 logically entails v2 (v1 → v2).

        A entails B iff every truth-assignment making A true also makes B true.
        In the hypercube: A → B iff A ≤ B component-wise (A is more specific).
        For continuous points: approximate with threshold.
        """
        return bool(np.all(v2 - v1 >= -threshold))

    def is_contradiction(self, v1: np.ndarray, v2: np.ndarray,
                          threshold: float = 0.1) -> bool:
        """
        Check if v1 and v2 are contradictory (cannot both be true).

        Contradiction = maximum Hamming distance from each other
        AND one is close to the negation of the other.
        """
        neg = self.negate(v1)
        return self.continuous_distance(v2, neg) < threshold

    def nearest_vertex(self, v: np.ndarray) -> np.ndarray:
        """
        Find the nearest {0,1} hypercube vertex to a continuous point v.

        This is the 'snap to logic' operation — finding the nearest
        crisp logical state.
        """
        return np.round(np.clip(v, 0.0, 1.0)).astype(np.float64)

    def interpolate(self, v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
        """
        Smooth interpolation between two logical points.

        t=0 → v1, t=1 → v2.
        Used by the flow engine to trace paths through logical space.
        """
        return (1.0 - t) * v1 + t * v2

    # ------------------------------------------------------------------ #
    # Topology helpers                                                       #
    # ------------------------------------------------------------------ #

    def contradiction_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return all pairs of maximally contradictory vertices.

        Maximum contradiction = Hamming distance of n_dims (all bits flipped).
        """
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        n = len(self.vertices)
        for i in range(n // 2):   # each pair appears once
            v = self.vertices[i]
            nv = self.negate(v)
            pairs.append((v, nv))
        return pairs

    def entailment_neighbours(self, v: np.ndarray) -> np.ndarray:
        """
        Return all vertices that v entails (v points toward).

        A vertex w is entailed by v iff v ≤ w component-wise.
        Returns array of shape (k, n_dims).
        """
        mask = np.all(self.vertices >= v - 0.01, axis=1)
        return self.vertices[mask]

    def logical_centre(self) -> np.ndarray:
        """
        The logical centre of the hypercube: (0.5, 0.5, ..., 0.5).

        This is the point of maximum ambiguity — equidistant from all
        crisp logical states.
        """
        return np.full(self.dim, 0.5)

    def uncertainty_score(self, v: np.ndarray) -> float:
        """
        Measure how far v is from the nearest vertex (0 = crisp, 1 = maximally uncertain).

        0.0 = perfectly crisp (on a vertex)
        1.0 = maximally uncertain (at the centroid)
        """
        nearest = self.nearest_vertex(v)
        max_d = np.linalg.norm(self.logical_centre() - np.zeros(self.dim))
        d = np.linalg.norm(v - nearest)
        return float(min(d / (max_d + 1e-12), 1.0))

    def summary(self) -> str:
        return (
            f"LogicalGeometry(n_dims={self.n_dims}, "
            f"n_vertices={len(self.vertices)}, "
            f"dim={self.dim})"
        )
