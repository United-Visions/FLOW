"""
Fiber Bundle Composer
=====================
Composes the four base geometries into a single unified seed manifold M₀
via fiber bundle construction.

Fiber bundle terminology
------------------------
A fiber bundle E is a space that locally looks like Base × Fiber:
  - Base B:    the similarity manifold (64D) — the "floor"
  - Fibers F:  attached to each point in B, one per geometry:
                 causal fiber    (16D) — the "flow"
                 logical fiber   ( 8D) — the "walls"
                 probabilistic fiber (16D) — the "light"

The total space E = B ⊕ F_causal ⊕ F_logical ⊕ F_prob has dimension:
  64 + 16 + 8 + 16 = 104

The projection map π: E → B forgets the fiber coordinates.
A point in E is (b, f_c, f_l, f_p) where b ∈ B and fₓ ∈ Fₓ.

Metric composition
------------------
The composed metric tensor is block-diagonal with inter-block coupling
terms.  The coupling terms encode how the fibers interact:
  - Causal–probabilistic coupling: certainty affects causal strength
  - Logical–probabilistic coupling: logical state affects confidence
  - All other cross-terms are zero in the seed (added by experience later)

Reference: Steenrod, "The Topology of Fibre Bundles" (1951)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .causal import CausalGeometry, DIM_CAUSAL
from .logical import LogicalGeometry, DIM_LOGICAL
from .probabilistic import ProbabilisticGeometry, DIM_PROB
from .similarity import SimilarityGeometry, DIM_SIMILARITY

# ── Bundle dimensions ──────────────────────────────────────────────────────
DIM_TOTAL    = DIM_SIMILARITY + DIM_CAUSAL + DIM_LOGICAL + DIM_PROB   # 104
SLICE_BASE   = slice(0, DIM_SIMILARITY)                                # 0:64
SLICE_CAUSAL = slice(DIM_SIMILARITY, DIM_SIMILARITY + DIM_CAUSAL)     # 64:80
SLICE_LOGICAL = slice(
    DIM_SIMILARITY + DIM_CAUSAL,
    DIM_SIMILARITY + DIM_CAUSAL + DIM_LOGICAL                         # 80:88
)
SLICE_PROB = slice(
    DIM_SIMILARITY + DIM_CAUSAL + DIM_LOGICAL,
    DIM_TOTAL                                                          # 88:104
)


@dataclass
class FiberSlices:
    """Convenience object: named slices into a DIM_TOTAL vector."""
    base:   slice = SLICE_BASE
    causal: slice = SLICE_CAUSAL
    logical: slice = SLICE_LOGICAL
    prob:   slice = SLICE_PROB
    total:  int   = DIM_TOTAL


SLICES = FiberSlices()


class FiberBundleComposer:
    """
    Composes four base geometries into the unified seed bundle M₀.

    Usage
    -----
    composer = FiberBundleComposer(sim, cau, log, prob)
    metric   = composer.compose_metric(point_104d)
    point    = composer.bundle_point(base_64d, causal_16d, logical_8d, prob_16d)
    base, fibers = composer.project(point_104d)
    """

    def __init__(
        self,
        sim:  SimilarityGeometry,
        cau:  CausalGeometry,
        log:  LogicalGeometry,
        prob: ProbabilisticGeometry,
    ) -> None:
        self.sim  = sim
        self.cau  = cau
        self.log  = log
        self.prob = prob
        self.slices = SLICES

    # ------------------------------------------------------------------ #
    # Point construction                                                    #
    # ------------------------------------------------------------------ #

    def bundle_point(
        self,
        base:    np.ndarray,
        causal:  np.ndarray,
        logical: np.ndarray,
        prob:    np.ndarray,
    ) -> np.ndarray:
        """
        Concatenate (base, causal, logical, prob) into a single 104D vector.

        Parameters
        ----------
        base    : (64,) base manifold coordinates
        causal  : (16,) causal fiber coordinates
        logical : (8,)  logical fiber coordinates
        prob    : (16,) probabilistic fiber coordinates (probability vector)

        Returns
        -------
        np.ndarray of shape (104,)
        """
        assert base.shape    == (DIM_SIMILARITY,), f"base must be ({DIM_SIMILARITY},), got {base.shape}"
        assert causal.shape  == (DIM_CAUSAL,),     f"causal must be ({DIM_CAUSAL},), got {causal.shape}"
        assert logical.shape == (DIM_LOGICAL,),    f"logical must be ({DIM_LOGICAL},), got {logical.shape}"
        assert prob.shape    == (DIM_PROB,),        f"prob must be ({DIM_PROB},), got {prob.shape}"
        return np.concatenate([base, causal, logical, prob])

    def project(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose a 104D point into its (base, causal, logical, prob) components.

        Returns
        -------
        (base_64, causal_16, logical_8, prob_16)
        """
        assert p.shape == (DIM_TOTAL,), f"Expected ({DIM_TOTAL},), got {p.shape}"
        return (
            p[SLICE_BASE].copy(),
            p[SLICE_CAUSAL].copy(),
            p[SLICE_LOGICAL].copy(),
            p[SLICE_PROB].copy(),
        )

    def base(self, p: np.ndarray) -> np.ndarray:
        """Extract the base manifold component (64D)."""
        return p[SLICE_BASE]

    def causal_fiber(self, p: np.ndarray) -> np.ndarray:
        """Extract the causal fiber component (16D)."""
        return p[SLICE_CAUSAL]

    def logical_fiber(self, p: np.ndarray) -> np.ndarray:
        """Extract the logical fiber component (8D)."""
        return p[SLICE_LOGICAL]

    def prob_fiber(self, p: np.ndarray) -> np.ndarray:
        """Extract the probabilistic fiber component (16D)."""
        return p[SLICE_PROB]

    # ------------------------------------------------------------------ #
    # Composed metric tensor                                                #
    # ------------------------------------------------------------------ #

    def compose_metric(self, p: np.ndarray) -> np.ndarray:
        """
        Compute the composed Riemannian metric tensor G(p) at point p.

        Structure
        ---------
        G is block-diagonal with inter-block coupling:

            G = [ G_base    0        0       0      ]
                [ 0       G_causal   0       C_{cp} ]
                [ 0         0      G_logical C_{lp} ]
                [ 0       C_{cp}ᵀ  C_{lp}ᵀ G_prob  ]

        Where:
          G_base    — similarity metric   (64×64)
          G_causal  — causal metric       (16×16)
          G_logical — logical metric      ( 8×8)
          G_prob    — Fisher metric       (16×16) at p's prob component
          C_{cp}    — causal–prob coupling (16×16)
          C_{lp}    — logical–prob coupling (8×16)

        Shape: (104, 104)
        """
        base_p    = p[SLICE_BASE]
        causal_p  = p[SLICE_CAUSAL]
        prob_p    = p[SLICE_PROB]

        G = np.zeros((DIM_TOTAL, DIM_TOTAL))

        # ── Block diagonal ─────────────────────────────────────────────
        # Base manifold metric
        G[SLICE_BASE, SLICE_BASE] = self.sim.metric_tensor(base_p)

        # Causal fiber metric
        G[SLICE_CAUSAL, SLICE_CAUSAL] = self.cau.causal_metric

        # Logical fiber metric
        G[SLICE_LOGICAL, SLICE_LOGICAL] = self.log.logical_metric

        # Probabilistic fiber metric (Fisher at current prob point)
        prob_valid = self.prob.normalize(prob_p)
        G[SLICE_PROB, SLICE_PROB] = self.prob.fisher_metric(prob_valid) * 0.01
        # Scale Fisher down — it can be very large for concentrated distributions

        # ── Inter-block coupling terms ─────────────────────────────────
        # Causal–prob coupling:
        # High confidence (low entropy) amplifies causal signal.
        # Coupling strength ∝ confidence.
        conf = self.prob.confidence(prob_valid)
        C_cp = 0.1 * conf * np.eye(DIM_CAUSAL, DIM_PROB)
        G[SLICE_CAUSAL, SLICE_PROB] = C_cp
        G[SLICE_PROB, SLICE_CAUSAL] = C_cp.T

        # Logical–prob coupling:
        # Logical certainty (vertex proximity) amplifies probabilistic signal.
        logical_p = p[SLICE_LOGICAL]
        logical_uncert = self.log.uncertainty_score(logical_p)
        C_lp = 0.05 * (1.0 - logical_uncert) * np.eye(DIM_LOGICAL, DIM_PROB)
        G[SLICE_LOGICAL, SLICE_PROB] = C_lp
        G[SLICE_PROB, SLICE_LOGICAL] = C_lp.T

        return G

    def bundle_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Composite Riemannian distance between two 104D points.

        Uses the midpoint metric (first-order approximation).
        """
        mid = (p1 + p2) / 2.0
        G   = self.compose_metric(mid)
        d   = p2 - p1
        d_sq = float(d @ G @ d)
        return float(np.sqrt(max(d_sq, 0.0)))

    # ------------------------------------------------------------------ #
    # Default / neutral fibers                                              #
    # ------------------------------------------------------------------ #

    def neutral_causal_fiber(self) -> np.ndarray:
        """
        A "neutral" causal fiber: no strong causal associations.
        All Pearl layers equally active, τ = 0.5.
        """
        v = np.zeros(DIM_CAUSAL)
        v[0] = 0.33; v[1] = 0.33; v[2] = 0.33   # equal layer activation
        v[3] = 0.5                                  # middle of causal time
        return v

    def neutral_logical_fiber(self) -> np.ndarray:
        """
        Neutral logical fiber: maximum ambiguity (centroid of hypercube).
        """
        return self.log.logical_centre()

    def neutral_prob_fiber(self) -> np.ndarray:
        """
        Neutral probabilistic fiber: uniform distribution (max uncertainty).
        """
        return self.prob.centroid.copy()

    def all_neutral(self) -> np.ndarray:
        """
        A completely neutral 104D point: no strong structure in any fiber.
        Used as an initialisation default.
        """
        base    = np.zeros(DIM_SIMILARITY)
        causal  = self.neutral_causal_fiber()
        logical = self.neutral_logical_fiber()
        prob    = self.neutral_prob_fiber()
        return self.bundle_point(base, causal, logical, prob)

    def validate_metric(self, p: np.ndarray) -> dict:
        """
        Validate that the composed metric at p is positive semi-definite.

        Returns a dict with validation results.
        """
        G = self.compose_metric(p)
        eigenvalues = np.linalg.eigvalsh(G)
        min_eig = float(eigenvalues.min())
        max_eig = float(eigenvalues.max())
        return {
            "is_symmetric": bool(np.allclose(G, G.T, atol=1e-10)),
            "is_psd": bool(min_eig >= -1e-8),
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": float(max_eig / max(abs(min_eig), 1e-12)),
        }

    def summary(self) -> str:
        return (
            f"FiberBundleComposer("
            f"total_dim={DIM_TOTAL}, "
            f"base={DIM_SIMILARITY}D, "
            f"causal={DIM_CAUSAL}D, "
            f"logical={DIM_LOGICAL}D, "
            f"prob={DIM_PROB}D)"
        )
