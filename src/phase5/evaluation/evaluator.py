"""PipelineEvaluator — evaluation harness for the full FLOW pipeline.

This evaluator provides a suite of geometry-grounded assessments that are
appropriate for a weight-free, token-free architecture.  Standard NLP
benchmarks (BLEU, ROUGE, perplexity) are intentionally absent: they assume
token distributions produced by weight matrices, neither of which exist here.

Evaluation tasks
----------------
1. evaluate_query(vector, label)
   → EvaluationResult with CoherenceMetrics for a single query.

2. evaluate_causal_direction(cause_vec, effect_vec)
   → CausalMetrics: does the flow respect causal order?

3. evaluate_novelty_decay(vectors, n_reps)
   → list[float]: does repeated exposure reduce novelty? (it must)

4. evaluate_locality(vector, label, radius)
   → LocalityMetrics: are deformations strictly local?

5. run_suite(vectors, labels)
   → SuiteResult: runs all four tasks and aggregates statistics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.phase3.annealing_engine.experience import Experience
from src.phase4.flow_engine.engine import FlowEngine
from src.phase4.flow_engine.query import Query

from ..pipeline.pipeline import GEOPipeline
from .metrics import (
    CoherenceMetrics,
    CausalMetrics,
    LocalityMetrics,
    EvaluationResult,
)
from .suite import SuiteResult


class PipelineEvaluator:
    """Runs geometry-grounded evaluations against a GEOPipeline instance.

    Parameters
    ----------
    pipeline : GEOPipeline
        The pipeline to evaluate.  May be pre-loaded with experiences.
    """

    def __init__(self, pipeline: GEOPipeline) -> None:
        self.pipeline = pipeline

    # ── 1. Single-query coherence ──────────────────────────────────────────

    def evaluate_query(
        self,
        vector: np.ndarray,
        label: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a single query end-to-end and return CoherenceMetrics.

        Parameters
        ----------
        vector : 104D numpy array
        label  : optional human-readable tag
        """
        result = self.pipeline.query(vector, label=label)
        coherence = CoherenceMetrics.from_result(
            result.trajectory, result.wave, result.output
        )
        return EvaluationResult(
            label=label or "unlabelled",
            coherence=coherence,
        )

    # ── 2. Causal direction ────────────────────────────────────────────────

    def evaluate_causal_direction(
        self,
        cause_vec: np.ndarray,
        effect_vec: np.ndarray,
        cause_label: str = "eval::cause",
        effect_label: str = "eval::effect",
    ) -> CausalMetrics:
        """Compare forward (cause→effect) vs backward (effect→cause) flow.

        A healthy causal geometry produces shorter / faster forward
        trajectories than backward ones.

        The two positions are temporarily placed on the manifold if they
        lack labels already; they are never removed (locality guarantee).
        """
        manifold = self.pipeline.manifold

        # Ensure both points exist on the manifold
        if cause_label not in manifold.labels:
            manifold.place(cause_label, cause_vec)
            manifold.update_density(cause_label)
        if effect_label not in manifold.labels:
            manifold.place(effect_label, effect_vec)
            manifold.update_density(effect_label)

        cause_pos  = manifold.position(cause_label)
        effect_pos = manifold.position(effect_label)

        # Build a fresh FlowEngine for each direction (same manifold, same dt)
        fe = FlowEngine(manifold, max_steps=100, dt=0.05, seed=0)

        fwd_q   = Query(vector=cause_pos.copy(),  label="cause→effect",
                        attractor_label=effect_label)
        bwd_q   = Query(vector=effect_pos.copy(), label="effect→cause",
                        attractor_label=cause_label)

        fwd_traj = fe.flow(fwd_q)
        bwd_traj = fe.flow(bwd_q)

        return CausalMetrics.from_trajectories(
            manifold=manifold,
            cause_pos=cause_pos,
            effect_pos=effect_pos,
            forward_traj=fwd_traj,
            backward_traj=bwd_traj,
        )

    # ── 3. Novelty decay ─────────────────────────────────────────────────

    def evaluate_novelty_decay(
        self,
        vector: np.ndarray,
        label: str = "novelty_eval::concept",
        n_reps: int = 5,
    ) -> List[float]:
        """Measure novelty scores over n_reps repeated exposures.

        Each call to learn() with the same experience should
        produce a lower novelty score, reflecting that the manifold
        has been shaped by the repeated exposure.

        Returns
        -------
        list of n_reps floats; should be monotonically decreasing.
        """
        novelties: List[float] = []
        for i in range(n_reps):
            exp = Experience(vector=vector.copy(), label=f"{label}_{i}")
            result = self.pipeline.learn(exp)
            novelties.append(float(result.novelty))
        return novelties

    # ── 4. Locality verification ──────────────────────────────────────────

    def evaluate_locality(
        self,
        vector: np.ndarray,
        label: str = "locality_eval::anchor",
    ) -> LocalityMetrics:
        """Verify that a deformation moves nearby points but not distant ones.

        Captures snapshots before and after applying a small deformation
        and classifies each moved point as near (within 3 × locality_radius)
        or far.  The locality radius is derived from the manifold density at
        the anchor, matching the actual Gaussian kernel used by deform_local().
        """
        manifold = self.pipeline.manifold
        all_labels = manifold.labels

        # Place the anchor if it is not already on the manifold
        if label not in all_labels:
            manifold.place(label, vector)
            manifold.update_density(label)
            all_labels = manifold.labels

        # Read the effective locality radius from the manifold density
        locality_radius = float(manifold.locality_radius(manifold.position(label)))

        # Snapshot: positions of all existing manifold points before deformation
        before: List[Tuple[str, np.ndarray]] = [
            (lbl, manifold.position(lbl).copy()) for lbl in all_labels
        ]

        displacement = np.zeros(manifold.DIM)
        displacement[0] = 0.1  # small perturbation in dim-0
        manifold.deform_local(label, displacement)

        # Snapshot after deformation
        after: List[Tuple[str, np.ndarray]] = [
            (lbl, manifold.position(lbl).copy()) for lbl in all_labels
        ]

        return LocalityMetrics.measure(
            manifold=manifold,
            anchor_label=label,
            snapshots_before=before,
            snapshots_after=after,
            locality_radius=locality_radius,
        )

    # ── 5. Full evaluation suite ──────────────────────────────────────────

    def run_suite(
        self,
        vectors: List[np.ndarray],
        labels: Optional[List[str]] = None,
        run_causal: bool = True,
        run_locality: bool = True,
        novelty_reps: int = 5,
    ) -> SuiteResult:
        """Run the complete evaluation suite over a list of query vectors.

        Parameters
        ----------
        vectors     : list of 104D numpy arrays (query meanings)
        labels      : optional list of string labels (same length as vectors)
        run_causal  : whether to run the causal direction evaluation
        run_locality: whether to run the locality check
        novelty_reps: number of repetitions for the novelty decay evaluation

        Returns
        -------
        SuiteResult with per-query EvaluationResults and aggregate stats.
        """
        if labels is None:
            labels = [f"query_{i}" for i in range(len(vectors))]

        # ── Per-query coherence evaluations ───────────────────────────────
        results: List[EvaluationResult] = []
        for vec, lbl in zip(vectors, labels):
            er = self.evaluate_query(vec, label=lbl)
            results.append(er)

        # ── Causal direction evaluation ────────────────────────────────────
        causal_metrics: Optional[CausalMetrics] = None
        if run_causal and len(vectors) >= 2:
            causal_metrics = self.evaluate_causal_direction(
                cause_vec=vectors[0],
                effect_vec=vectors[-1],
                cause_label="suite_eval::cause",
                effect_label="suite_eval::effect",
            )

        # ── Locality check ─────────────────────────────────────────────────
        locality_metrics: Optional[LocalityMetrics] = None
        if run_locality and len(vectors) >= 1:
            locality_metrics = self.evaluate_locality(
                vector=vectors[0],
                label="suite_eval::locality_anchor",
            )

        # ── Novelty decay (first vector, repeated exposure) ───────────────
        novelty_scores: List[float] = []
        if len(vectors) >= 1:
            novelty_scores = self.evaluate_novelty_decay(
                vector=vectors[0],
                label="suite_eval::novelty",
                n_reps=novelty_reps,
            )

        return SuiteResult(
            results=results,
            causal=causal_metrics,
            locality=locality_metrics,
            novelty_decay=novelty_scores,
        )
