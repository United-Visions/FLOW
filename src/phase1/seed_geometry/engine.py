"""
Seed Geometry Engine — Component 1 (Public API)
================================================
The single public entry point for Phase 1a.

Responsibility
--------------
Compute the mathematical skeleton of the manifold from first principles.
Runs exactly once.  Output M₀ is static forever.

Usage
-----
    from src.phase1.seed_geometry import SeedGeometryEngine

    engine = SeedGeometryEngine()
    M0 = engine.build()

    # Query the seed manifold
    p = M0.seed_points[0]
    q = M0.seed_points[1]
    d = M0.distance(p, q)
    v = M0.validate()

What this engine does
---------------------
1. Build CausalGeometry          from Pearl's do-calculus
2. Build LogicalGeometry         from Boolean algebra
3. Build ProbabilisticGeometry   from Kolmogorov axioms + Fisher metric
4. Build SimilarityGeometry      from metric space axioms
5. Compose them via FiberBundleComposer into a unified 104D bundle
6. Generate the archetypal seed points (the manifold skeleton)
7. Return M₀ — the immutable SeedManifold

Design constraint satisfied
---------------------------
  NO WEIGHTS  — no numerical parameters tuned here
  NO DATA     — zero data points required
  ONE-TIME    — this engine runs once; its output never changes
"""

from __future__ import annotations

import time
import numpy as np
from typing import List

from .causal import CausalGeometry, DIM_CAUSAL
from .logical import LogicalGeometry, DIM_LOGICAL, N_LOGICAL_DIMS
from .probabilistic import ProbabilisticGeometry, DIM_PROB
from .similarity import SimilarityGeometry, DIM_SIMILARITY, DOMAIN_TAXONOMY
from .composer import FiberBundleComposer, DIM_TOTAL, SLICES
from .manifold import SeedManifold, ManifoldPoint


class SeedGeometryEngine:
    """
    Derives and composes the four base geometries into the seed manifold M₀.

    This component runs once.  Its output is static forever.

    Conformance
    -----------
    - Takes no external data (satisfies Design Principle 1, 3)
    - Produces M₀ = the initial state for Component 2 (Living Manifold)
    - M₀ encodes causality, logic, probability, and similarity without
      a single training example
    """

    def __init__(self) -> None:
        self._built: bool = False
        self._M0: SeedManifold | None = None

    # ------------------------------------------------------------------ #
    # Main API                                                              #
    # ------------------------------------------------------------------ #

    def build(self) -> SeedManifold:
        """
        Build M₀ from first principles.

        Steps
        -----
        1. Derive four base geometries
        2. Compose via fiber bundle
        3. Generate archetypal seed points from each geometry
        4. Assemble SeedManifold
        5. Validate and return

        Returns
        -------
        SeedManifold
            The immutable seed manifold M₀.

        Raises
        ------
        RuntimeError
            If validation fails — the resulting manifold is not geometrically sound.
        """
        if self._built and self._M0 is not None:
            return self._M0   # idempotent

        t_start = time.perf_counter()

        # ── Step 1: Derive base geometries ────────────────────────────
        print("  [1/5] Deriving causal geometry from Pearl's do-calculus...")
        cau  = CausalGeometry.build()

        print("  [2/5] Deriving logical geometry from Boolean algebra...")
        log  = LogicalGeometry.build()

        print("  [3/5] Deriving probabilistic geometry from Kolmogorov axioms...")
        prob = ProbabilisticGeometry.build()

        print("  [4/5] Deriving similarity geometry from metric space axioms...")
        sim  = SimilarityGeometry.build()

        # ── Step 2: Compose via fiber bundle ──────────────────────────
        print("  [5/5] Composing into unified bundle via fiber bundle construction...")
        composer = FiberBundleComposer(sim, cau, log, prob)

        # ── Step 3: Generate archetypal seed points ───────────────────
        seed_points = self._generate_seed_points(sim, cau, log, prob, composer)

        # ── Step 4: Assemble SeedManifold ─────────────────────────────
        build_time = time.perf_counter() - t_start
        M0 = SeedManifold(
            sim=sim, cau=cau, log=log, prob=prob,
            composer=composer,
            seed_points=seed_points,
            build_time_s=build_time,
        )

        # ── Step 5: Validate ─────────────────────────────────────────
        validation = M0.validate()
        self._check_validation(validation)

        self._built = True
        self._M0    = M0

        print(f"\nM₀ built successfully in {build_time:.3f}s")
        print(M0.summary())
        return M0

    # ------------------------------------------------------------------ #
    # Seed point generation                                                 #
    # ------------------------------------------------------------------ #

    def _generate_seed_points(
        self,
        sim:      SimilarityGeometry,
        cau:      CausalGeometry,
        log:      LogicalGeometry,
        prob:     ProbabilisticGeometry,
        composer: FiberBundleComposer,
    ) -> List[ManifoldPoint]:
        """
        Generate the archetypal seed points that form the manifold skeleton.

        Four classes of seed points
        ----------------------------
        A. Causal archetypes     — one per causal DAG node
        B. Logical archetypes    — one per corner of hypercube (sampled)
        C. Probabilistic archetypes — vertices + centroid of simplex
        D. Domain centroids      — one per semantic domain in similarity geometry
        """
        points: List[ManifoldPoint] = []

        # ── A. Causal archetype points ────────────────────────────────
        for node_name, embedding in cau.embeddings.items():
            # Base: project causal embedding onto similarity space (first 64 dims via padding)
            base = np.zeros(DIM_SIMILARITY)
            # Place causal archetypes in the "causal_mechanisms" domain region
            causal_center = sim.domain_centers.get("causal_mechanisms", np.zeros(DIM_SIMILARITY))
            noise = np.random.default_rng(abs(hash(node_name)) % 2**32).normal(0, 0.05, DIM_SIMILARITY)
            base = causal_center + noise

            logical = log.logical_centre()     # neutral logic
            p_fib   = prob.centroid.copy()      # neutral probability

            vec = composer.bundle_point(base, embedding, logical, p_fib)
            points.append(ManifoldPoint(
                vector=vec,
                label=f"causal::{node_name}",
                origin="causal"
            ))

        # ── B. Logical archetype points (sampled hypercube corners) ───
        # Sample 2^4 = 16 representative corners from the 2^8 = 256 total
        rng = np.random.default_rng(42)
        logical_sample_indices = rng.choice(len(log.vertices), size=min(32, len(log.vertices)), replace=False)
        logic_center = sim.domain_centers.get("logical_entities", np.zeros(DIM_SIMILARITY))

        for idx in logical_sample_indices:
            vertex = log.vertices[idx]
            base = logic_center + rng.normal(0, 0.05, DIM_SIMILARITY)
            causal = composer.neutral_causal_fiber()

            # Probabilistic fiber reflects logical certainty:
            # crisp vertex → confident; centroid → uncertain
            certainty = 1.0 - log.uncertainty_score(vertex)
            # Concentrate probability on one outcome proportional to certainty
            p_fib = np.full(DIM_PROB, (1.0 - certainty) / max(DIM_PROB - 1, 1))
            p_fib[0] += certainty
            p_fib = prob.normalize(p_fib)

            vec = composer.bundle_point(base, causal, vertex, p_fib)
            bits_str = "".join(str(int(b)) for b in vertex)
            points.append(ManifoldPoint(
                vector=vec,
                label=f"logical::{bits_str}",
                origin="logical"
            ))

        # ── C. Probabilistic archetype points ─────────────────────────
        prob_center = sim.domain_centers.get("epistemic", np.zeros(DIM_SIMILARITY))

        # Add the k certainty vertices (one concept is 100% likely)
        for i in range(prob.k):
            vertex = prob.vertices[i]
            base   = prob_center + rng.normal(0, 0.05, DIM_SIMILARITY)
            causal = composer.neutral_causal_fiber()
            logical = log.logical_centre()
            vec = composer.bundle_point(base, causal, logical, vertex)
            points.append(ManifoldPoint(
                vector=vec,
                label=f"prob::certain_{i}",
                origin="probabilistic"
            ))

        # Add the centroid (maximum uncertainty)
        base = prob_center + rng.normal(0, 0.05, DIM_SIMILARITY)
        vec  = composer.bundle_point(base, composer.neutral_causal_fiber(),
                                     log.logical_centre(), prob.centroid)
        points.append(ManifoldPoint(
            vector=vec,
            label="prob::maximal_uncertainty",
            origin="probabilistic"
        ))

        # ── D. Domain centroid points ─────────────────────────────────
        for domain_name, center_vec in sim.domain_centers.items():
            causal  = composer.neutral_causal_fiber()
            logical = log.logical_centre()
            p_fib   = prob.centroid.copy()
            vec = composer.bundle_point(center_vec, causal, logical, p_fib)
            points.append(ManifoldPoint(
                vector=vec,
                label=f"domain::{domain_name}",
                origin="similarity"
            ))

        return points

    # ------------------------------------------------------------------ #
    # Validation                                                            #
    # ------------------------------------------------------------------ #

    def _check_validation(self, results: dict) -> None:
        """
        Raise if any critical validation checks fail.
        """
        errors = []
        if not results.get("all_points_correct_dim", True):
            errors.append(f"Points with wrong dimensionality: {results.get('wrong_dim_labels')}")
        if not results.get("metric_psd", True):
            errors.append(f"Metric is not PSD at neutral point. Min eigenvalue: {results.get('metric_min_eigenvalue')}")
        if not results.get("metric_symmetric", True):
            errors.append("Metric tensor is not symmetric.")
        if "triangle_inequality" in results and not results["triangle_inequality"]:
            errors.append("Triangle inequality violated.")
        if "distances_non_negative" in results and not results["distances_non_negative"]:
            errors.append("Negative distances detected.")

        if errors:
            raise RuntimeError(
                "SeedGeometryEngine: M₀ validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    @property
    def is_built(self) -> bool:
        """True if M₀ has been successfully built."""
        return self._built
