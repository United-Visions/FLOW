"""
Probabilistic Geometry
======================
Derives probabilistic structure from first principles.

Source:   Kolmogorov axioms + information geometry (Amari)
Geometry: Statistical simplex with Fisher-Rao metric.

Properties encoded
------------------
- Certainty     : vertices of the simplex (one outcome certain)
- Uncertainty   : interior of the simplex (diffuse probability mass)
- Confidence    : distance from centroid toward nearest vertex
- Probability flow : natural geodesics along the Fisher metric
- Distance      : KL divergence as natural dissimilarity

Design decision
---------------
We use a K-dimensional simplex where K = DIM_PROB is the number of
distinct epistemic states the geometry must represent.  The Fisher
metric makes geodesics physically meaningful — they are natural 
probability interpolation paths, not arbitrary straight lines.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Dimensionality of the probabilistic fiber
# The simplex lives in DIM_PROB - 1 effective dimensions
DIM_PROB = 16  # K distinct outcome categories
EPS = 1e-12    # numerical floor to avoid log(0)


@dataclass
class ProbabilisticGeometry:
    """
    The probabilistic geometric fiber derived from Kolmogorov axioms
    and information geometry.

    Attributes
    ----------
    k : int
        Number of probability categories (simplex dimension = k - 1).
    dim : int
        Ambient dimensionality of this fiber (= k).
    vertices : np.ndarray
        Shape (k, k).  Each row is a standard basis vector eᵢ (certainty of i).
    centroid : np.ndarray
        The uniform distribution (1/k, ..., 1/k) — maximum uncertainty.
    """

    k: int = DIM_PROB
    dim: int = DIM_PROB
    vertices: np.ndarray = field(default_factory=lambda: np.eye(DIM_PROB))
    centroid: np.ndarray = field(
        default_factory=lambda: np.full(DIM_PROB, 1.0 / DIM_PROB)
    )

    # ------------------------------------------------------------------ #
    # Construction                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls) -> "ProbabilisticGeometry":
        """Derive the probabilistic geometry from the Kolmogorov axioms."""
        geo = cls()
        geo.vertices = np.eye(geo.k)
        geo.centroid = np.full(geo.k, 1.0 / geo.k)
        return geo

    # ------------------------------------------------------------------ #
    # Projection and normalisation                                          #
    # ------------------------------------------------------------------ #

    def to_simplex(self, v: np.ndarray) -> np.ndarray:
        """
        Project a vector onto the probability simplex.

        Output is a valid probability distribution: non-negative entries
        summing to 1.  Uses the efficient Duchi et al. (2008) algorithm.
        """
        v = np.asarray(v, dtype=np.float64)
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0][-1]
        theta = float((cssv[rho] - 1.0) / (rho + 1.0))
        return np.maximum(v - theta, 0.0)

    def normalize(self, v: np.ndarray) -> np.ndarray:
        """Ensure v is a valid probability vector (positive, sums to 1)."""
        v = np.clip(v, EPS, None)
        return v / v.sum()

    # ------------------------------------------------------------------ #
    # Fisher metric and information-geometric distances                     #
    # ------------------------------------------------------------------ #

    def fisher_metric(self, p: np.ndarray) -> np.ndarray:
        """
        Fisher information matrix at probability vector p.

        For a categorical distribution over K outcomes:
            g_ij(p) = δ_ij / p_i   (diagonal matrix)

        This is the unique metric (up to scaling) consistent with the
        statistical structure of the simplex (Chentsov's theorem).

        Returns
        -------
        np.ndarray, shape (k, k)
            Diagonal Fisher information matrix at p.
        """
        p = self.normalize(p)
        return np.diag(1.0 / p)

    def riemannian_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Geodesic distance under the Fisher-Rao metric (Bhattacharyya).

        The exact Fisher-Rao geodesic distance on the simplex equals:
            d_FR(p, q) = 2 · arccos(Σᵢ √(pᵢ qᵢ))

        This is the Riemannian distance on the positive orthant of the
        (k-1)-sphere under the Fisher metric.

        Domain: p, q ∈ Δ^(k-1)  (the probability simplex)
        Range:  [0, π]
        """
        p = self.normalize(p)
        q = self.normalize(q)
        bc = float(np.sum(np.sqrt(p * q)))            # Bhattacharyya coefficient
        bc = np.clip(bc, -1.0 + EPS, 1.0 - EPS)
        return 2.0 * float(np.arccos(bc))

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Kullback-Leibler divergence D_KL(p || q).

        Not a metric (asymmetric), but the natural divergence on the
        statistical manifold.  Used as an asymmetric dissimilarity.

        D_KL(p || q) = Σᵢ pᵢ log(pᵢ / qᵢ)
        """
        p = self.normalize(p)
        q = self.normalize(q)
        return float(np.sum(p * (np.log(p + EPS) - np.log(q + EPS))))

    def js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Jensen-Shannon divergence (symmetric, bounded).

        JSD(p, q) = (KL(p||m) + KL(q||m)) / 2   where m = (p + q) / 2.
        Range: [0, log(2)] ≈ [0, 0.693].
        """
        m = self.normalize((p + q) / 2.0)
        return (self.kl_divergence(p, m) + self.kl_divergence(q, m)) / 2.0

    # ------------------------------------------------------------------ #
    # Confidence and uncertainty                                            #
    # ------------------------------------------------------------------ #

    def entropy(self, p: np.ndarray) -> float:
        """
        Shannon entropy H(p) = -Σᵢ pᵢ log pᵢ.

        0 = maximum certainty (on a vertex).
        log(k) = maximum uncertainty (at centroid).
        """
        p = self.normalize(p)
        return float(-np.sum(p * np.log(p + EPS)))

    def confidence(self, p: np.ndarray) -> float:
        """
        Confidence score: how concentrated the probability mass is.

        Returns a value in [0, 1] where:
        0 = maximum uncertainty (uniform distribution, centroid)
        1 = maximum certainty (mass on one vertex)
        """
        h = self.entropy(p)
        h_max = np.log(self.k + EPS)
        return float(1.0 - h / h_max)

    def uncertainty_gradient(self, p: np.ndarray) -> np.ndarray:
        """
        Direction of increasing uncertainty at point p.

        The gradient of entropy ∇H(p) points toward the centroid.
        Used by the Annealing and Flow Engines to model epistemic drift.
        """
        p = self.normalize(p)
        # ∇H(p) = -(1 + log p)
        grad = -(1.0 + np.log(p + EPS))
        # Normalise to unit length
        norm = np.linalg.norm(grad)
        if norm < EPS:
            return np.zeros(self.k)
        return grad / norm

    # ------------------------------------------------------------------ #
    # Interpolation and geodesics                                           #
    # ------------------------------------------------------------------ #

    def geodesic(self, p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
        """
        Geodesic interpolation on the statistical simplex under Fisher metric.

        Uses the spherical interpolation (slerp) on the positive orthant
        of the unit sphere (√p and √q are unit vectors in the Fisher metric).

        t=0 → p, t=1 → q.
        """
        p = self.normalize(p)
        q = self.normalize(q)
        sp = np.sqrt(p)
        sq = np.sqrt(q)

        # Angle between them
        cos_theta = np.clip(float(np.dot(sp, sq)), -1.0 + EPS, 1.0 - EPS)
        theta = np.arccos(cos_theta)

        if theta < EPS:
            return p

        # Spherical interpolation in √-space
        slerp = (np.sin((1 - t) * theta) * sp + np.sin(t * theta) * sq) / np.sin(theta)
        interpolated = slerp ** 2
        return self.normalize(interpolated)

    def natural_gradient(self, p: np.ndarray, euclidean_grad: np.ndarray) -> np.ndarray:
        """
        Convert a Euclidean gradient to the natural gradient (Fisher metric).

        In information geometry, the natural gradient is:
            g̃ = G(p)^{-1} g   where G(p) is the Fisher metric

        For the diagonal Fisher metric this is:
            g̃ᵢ = pᵢ · gᵢ
        """
        p = self.normalize(p)
        return p * euclidean_grad

    # ------------------------------------------------------------------ #
    # Simplex topology                                                       #
    # ------------------------------------------------------------------ #

    def nearest_vertex(self, p: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Find the nearest certainty vertex (argmax outcome).

        Returns (vertex_index, vertex_vector).
        """
        idx = int(np.argmax(p))
        return idx, self.vertices[idx].copy()

    def center_of_mass(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Weighted Fréchet mean on the simplex (under Fisher metric).

        Approximated via weighted average on √-sphere, then squaring back.

        points  : (n, k) array of probability vectors
        weights : (n,) array of non-negative weights
        """
        weights = weights / (weights.sum() + EPS)
        sq_points = np.sqrt(np.clip(points, EPS, None))
        mean_sq = np.average(sq_points, axis=0, weights=weights)
        norm = np.linalg.norm(mean_sq)
        if norm < EPS:
            return self.centroid.copy()
        return self.normalize(mean_sq ** 2)

    def summary(self) -> str:
        return (
            f"ProbabilisticGeometry(k={self.k}, "
            f"dim={self.dim}, "
            f"max_uncertainty={np.log(self.k):.3f} nats)"
        )
