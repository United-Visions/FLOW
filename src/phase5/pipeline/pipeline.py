"""GEOPipeline — full end-to-end geometric reasoning pipeline (C1 → C7).

Wires all seven components defined in the Geometric Causal Architecture:

  Group A — The Manifold (What The System Knows)
    C1  SeedGeometryEngine   — derives M₀ once, static forever
    C2  LivingManifold       — dynamic M(t), READ + WRITE API
    C3  AnnealingEngine      — continuous self-organisation from raw experience
    C4  ContrastEngine       — SAME / DIFFERENT relational geometry shaping

  Group B — The Reasoner (What The System Does)
    C5  FlowEngine           — SDE navigation → Trajectory
    C6  ResonanceLayer       — Trajectory → StandingWave Ψ
    C7  ExpressionRenderer   — Ψ → natural language (real wave, not mock)

Design constraints upheld
--------------------------
NO WEIGHTS       — no tunable parameter matrices anywhere
NO TOKENS        — queries and results are continuous geometric objects
NO TRAINING      — the pipeline grows via learn() / contrast() at runtime
LOCAL UPDATES    — all manifold writes honour the Gaussian locality radius
CAUSALITY FIRST  — causal fiber is structurally encoded in every query path
SEPARATION       — each component is only invoked for its single role
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from src.phase1.seed_geometry.engine import SeedGeometryEngine
from src.phase1.expression.renderer import ExpressionRenderer
from src.phase2.living_manifold.manifold import LivingManifold
from src.phase2.contrast_engine.engine import ContrastEngine, ContrastResult, JudgmentType
from src.phase3.annealing_engine.engine import AnnealingEngine, AnnealingStats
from src.phase3.annealing_engine.experience import Experience, ExperienceResult
from src.phase4.flow_engine.engine import FlowEngine
from src.phase4.flow_engine.query import Query, Trajectory
from src.phase4.resonance_layer.layer import ResonanceLayer
from src.phase1.expression.wave import StandingWave
from src.persistence.snapshot import ManifoldSnapshot

from .result import PipelineResult


class GEOPipeline:
    """Full Geometric Causal Architecture pipeline: C1 → C7.

    Parameters
    ----------
    T0          : initial annealing temperature (default 1.0)
    lambda_     : cooling rate for T(t) = T₀·e^(−λt) + T_floor (default 0.01)
    T_floor     : minimum temperature — keeps unknown territory flexible (default 0.05)
    flow_max_steps  : maximum SDE steps per query (default 150)
    flow_dt         : SDE integration time step (default 0.05)
    flow_seed       : RNG seed for the Flow Engine (reproducible queries; None = random)

    Usage
    -----
    >>> pipeline = GEOPipeline()
    >>> pipeline.learn(Experience(vector=vec, label="concept::gravity"))
    >>> result  = pipeline.query(vec, label="what is gravity?")
    >>> print(result.text)
    """

    def __init__(
        self,
        T0: float = 1.0,
        lambda_: float = 0.01,
        T_floor: float = 0.05,
        flow_max_steps: int = 150,
        flow_dt: float = 0.05,
        flow_seed: Optional[int] = None,
    ) -> None:
        # ── C1 — build seed manifold once ─────────────────────────────────
        _seed_engine = SeedGeometryEngine()
        M0 = _seed_engine.build()

        # ── C2 — living manifold wrapping M₀ ──────────────────────────────
        self.manifold = LivingManifold(M0)

        # ── C3 — annealing engine ──────────────────────────────────────────
        self._annealing = AnnealingEngine(
            self.manifold, T0=T0, lambda_=lambda_, T_floor=T_floor
        )

        # ── C4 — contrast engine ───────────────────────────────────────────
        self._contrast_engine = ContrastEngine(self.manifold)

        # ── C5 — flow engine ───────────────────────────────────────────────
        self._flow_engine = FlowEngine(
            self.manifold,
            max_steps=flow_max_steps,
            dt=flow_dt,
            seed=flow_seed,
        )

        # ── C6 — resonance layer ───────────────────────────────────────────
        self._resonance = ResonanceLayer(self.manifold)

        # ── C7 — expression renderer (receives real Ψ from C6) ────────────
        self._dim: int = self.manifold.DIM
        self._renderer = ExpressionRenderer(dim=self._dim)

        # ── Internal bookkeeping ───────────────────────────────────────────
        self._query_count: int = 0

    # ── Group A: manifold shaping ops ─────────────────────────────────────

    def learn(self, experience: Experience) -> ExperienceResult:
        """C3 — Process a raw experience through the Annealing Engine.

        The experience is located in M(t), its novelty scored, and a
        Gaussian-weighted local deformation is applied.  This is the
        continuous growth operation — there is no separate training.

        Parameters
        ----------
        experience : Experience(vector, label=None, source=None)

        Returns
        -------
        ExperienceResult with novelty, delta_magnitude, n_affected, etc.
        """
        return self._annealing.process(experience)

    def learn_batch(self, experiences: List[Experience]) -> List[ExperienceResult]:
        """C3 — Process a list of experiences in sequence via the Annealing Engine."""
        return self._annealing.process_batch(experiences)

    def contrast(
        self,
        label_a: str,
        label_b: str,
        relation: str,
        strength: float = 1.0,
    ) -> ContrastResult:
        """C4 — Apply a same/different relational judgment between two concepts.

        Parameters
        ----------
        label_a, label_b : concept labels already placed on the manifold
        relation         : "same" or "different"
        strength         : scaling of the α/β displacement (default 1.0)

        Returns
        -------
        ContrastResult with delta_distance and displacement vectors.
        """
        judgment = JudgmentType(relation.lower())
        return self._contrast_engine.judge(label_a, label_b, judgment, strength=strength)

    # ── Group B: reasoning ops ─────────────────────────────────────────────

    def query(
        self,
        vector: np.ndarray,
        label: Optional[str] = None,
        attractor_label: Optional[str] = None,
    ) -> PipelineResult:
        """C5 → C6 → C7 — Full reasoning and rendering pipeline.

        1. C5 (FlowEngine) navigates M(t) from the query position via SDE,
           producing a continuous Trajectory.
        2. C6 (ResonanceLayer) accumulates the Trajectory into the
           pre-linguistic standing wave Ψ.
        3. C7 (ExpressionRenderer) renders Ψ into fluent natural language.

        Parameters
        ----------
        vector          : 104D query vector (the meaning of the question)
        label           : optional human-readable label for the query
        attractor_label : optional manifold label to use as the flow attractor

        Returns
        -------
        PipelineResult with .text, .confidence, .trajectory, .wave, etc.
        """
        self._query_count += 1

        q = Query(vector=vector, label=label, attractor_label=attractor_label)
        trajectory: Trajectory = self._flow_engine.flow(q)
        wave: StandingWave = self._resonance.accumulate(trajectory)
        output = self._renderer.render(wave)

        return PipelineResult(
            query=q,
            trajectory=trajectory,
            wave=wave,
            output=output,
        )

    # ── Introspection ──────────────────────────────────────────────────────

    @property
    def temperature(self) -> float:
        """Current annealing temperature T(t)."""
        return self._annealing.temperature

    @property
    def stats(self) -> AnnealingStats:
        """Running AnnealingStats from the annealing engine."""
        return self._annealing.stats

    @property
    def query_count(self) -> int:
        """Total number of queries processed by this pipeline instance."""
        return self._query_count

    @property
    def dimension(self) -> int:
        """Manifold dimensionality (104D)."""
        return self._dim

    @property
    def n_concepts(self) -> int:
        """Number of concept points currently on the manifold."""
        return len(self.manifold.labels)

    def reset_temperature(self) -> None:
        """Restart the annealing temperature schedule from T₀.

        This does NOT undo any deformations already applied to the manifold.
        """
        self._annealing.reset_temperature()

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str, vocabulary_path: Optional[str] = None) -> dict:
        """Save the pipeline state (manifold + optional vocabulary) to disk.

        Parameters
        ----------
        path : str
            File path for the manifold snapshot (.npz).
        vocabulary_path : str | None
            If provided, also save the current C7 vocabulary entries here.

        Returns
        -------
        dict with keys: n_points, manifold_path, vocabulary_path (if saved).
        """
        n = ManifoldSnapshot.save(self.manifold, path)
        result = {"n_points": n, "manifold_path": path}

        if vocabulary_path is not None:
            from src.vocabulary.vocabulary_store import VocabularyStore
            vocab = self._renderer.matcher.vocabulary
            if vocab:
                VocabularyStore.save(vocab, vocabulary_path)
                result["vocabulary_path"] = vocabulary_path
                result["n_vocab"] = len(vocab)

        return result

    @classmethod
    def load(
        cls,
        path: str,
        vocabulary_path: Optional[str] = None,
        T0: float = 1.0,
        lambda_: float = 0.01,
        T_floor: float = 0.05,
        flow_max_steps: int = 150,
        flow_dt: float = 0.05,
        flow_seed: Optional[int] = None,
    ) -> "GEOPipeline":
        """Load a pipeline from a saved manifold snapshot.

        Parameters
        ----------
        path : str
            Path to a .npz manifold snapshot created by ``save()``.
        vocabulary_path : str | None
            Optional path to a vocabulary .npz to load into C7.
        T0, lambda_, T_floor, flow_max_steps, flow_dt, flow_seed :
            Pipeline hyperparameters (same as ``__init__``).

        Returns
        -------
        GEOPipeline with restored manifold state.
        """
        pipeline = cls(
            T0=T0,
            lambda_=lambda_,
            T_floor=T_floor,
            flow_max_steps=flow_max_steps,
            flow_dt=flow_dt,
            flow_seed=flow_seed,
        )

        # Restore manifold state from snapshot
        ManifoldSnapshot.load(path, manifold=pipeline.manifold)

        # Optionally load vocabulary into C7
        if vocabulary_path is not None and os.path.exists(vocabulary_path):
            pipeline._renderer.matcher.load_vocabulary(vocabulary_path)

        return pipeline

    def summary(self) -> str:
        """Return a human-readable summary of pipeline state."""
        lines = [
            "GEOPipeline State",
            f"  Dimension        : {self.dimension}",
            f"  Concepts on M(t) : {self.n_concepts}",
            f"  Temperature T(t) : {self.temperature:.4f}",
            f"  Queries issued   : {self._query_count}",
            f"  Experiences seen : {self.stats.n_processed}",
            self.manifold.summary(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GEOPipeline(dim={self.dimension}, "
            f"concepts={self.n_concepts}, "
            f"T={self.temperature:.3f}, "
            f"queries={self._query_count})"
        )
