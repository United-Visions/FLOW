"""Evaluation metrics for the FLOW geometric pipeline.

Existing NLP benchmarks (BLEU, ROUGE, perplexity, …) are inappropriate for
this architecture — they all assume tokens and weight-trained probability
distributions.  This module provides metrics grounded in the geometric and
thermodynamic properties of the system itself.

The three metric groups are:

  CoherenceMetrics
    Measures how well-formed and confident the C6 → C7 output is.
    Purely intrinsic: uses wave confidence, render confidence, and
    the fraction of the wave that sits in crystallised (high-density) regions.

  CausalMetrics
    Measures whether the trajectory respects causal structure.
    Compares flow along cause→effect vs effect→cause using the causal
    fiber (dims 64–79) and C2's causal_direction() API.

  LocalityMetrics
    Verifies the Local-Updates constraint: after applying an experience,
    only nearby geometry moves; distant points are unaffected.
    This is a hard architectural guarantee, not an accuracy metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.phase2.living_manifold.manifold import LivingManifold
from src.phase4.flow_engine.query import Trajectory
from src.phase1.expression.wave import StandingWave
from src.phase1.expression.renderer import RenderedOutput


# ── CoherenceMetrics ─────────────────────────────────────────────────────────

@dataclass
class CoherenceMetrics:
    """Intrinsic quality metrics for a single query/response cycle.

    All values are in [0, 1] (higher = better) unless documented otherwise.

    Attributes
    ----------
    wave_confidence         : overall standing-wave confidence from C6
    render_confidence       : expression-match quality from C7
    flow_preserved          : True if C7 preserved the flow dynamics
    n_wave_points           : total number of wave points (Ψ > 0)
    n_core_wave_points      : wave points with normalised amplitude ≥ 0.4
    core_fraction           : n_core / n_total  (focussed = higher)
    mean_amplitude          : mean amplitude across all wave points
    trajectory_steps        : number of SDE integration steps (C5)
    trajectory_mean_speed   : mean speed of the flow trajectory
    trajectory_mean_curv    : mean curvature of the flow trajectory
    termination_reason      : why the flow stopped
    """

    wave_confidence: float
    render_confidence: float
    flow_preserved: bool
    n_wave_points: int
    n_core_wave_points: int
    core_fraction: float
    mean_amplitude: float
    trajectory_steps: int
    trajectory_mean_speed: float
    trajectory_mean_curv: float
    termination_reason: str

    @classmethod
    def from_result(cls, trajectory: Trajectory, wave: StandingWave,
                    output: RenderedOutput) -> "CoherenceMetrics":
        """Compute metrics from the raw C5 / C6 / C7 artefacts."""
        amplitudes = np.array([p.amplitude for p in wave.points]) if wave.points else np.array([0.0])
        n_total = len(wave.points)
        n_core = int(np.sum(amplitudes >= 0.4)) if n_total > 0 else 0
        core_fraction = n_core / n_total if n_total > 0 else 0.0
        mean_amp = float(np.mean(amplitudes)) if n_total > 0 else 0.0

        return cls(
            wave_confidence=float(wave.mean_confidence()),
            render_confidence=float(output.confidence),
            flow_preserved=bool(output.flow_preserved),
            n_wave_points=n_total,
            n_core_wave_points=n_core,
            core_fraction=core_fraction,
            mean_amplitude=mean_amp,
            trajectory_steps=len(trajectory),
            trajectory_mean_speed=float(trajectory.mean_speed),
            trajectory_mean_curv=float(trajectory.mean_curvature),
            termination_reason=trajectory.termination_reason,
        )

    def overall_score(self) -> float:
        """Aggregate coherence score ∈ [0, 1].

        Weighted combination:
          0.35 · render_confidence   (language quality)
          0.25 · wave_confidence     (wave quality)
          0.20 · core_fraction       (focussed meaning)
          0.20 · mean_amplitude      (wave energy)
        """
        return (
            0.35 * self.render_confidence
            + 0.25 * self.wave_confidence
            + 0.20 * self.core_fraction
            + 0.20 * self.mean_amplitude
        )


# ── CausalMetrics ─────────────────────────────────────────────────────────────

@dataclass
class CausalMetrics:
    """Measures whether the flow respects the causal fiber structure.

    A well-functioning system should navigate cause→effect more fluently
    than effect→cause: the trajectory should be shorter, faster, and
    more direct when following causal order.

    Attributes
    ----------
    causal_score        : advantage of forward (C→E) over backward (E→C) flow ∈ [0,1]
    forward_steps       : steps taken for cause → effect query
    backward_steps      : steps taken for effect → cause query
    forward_speed       : mean speed for cause → effect
    backward_speed      : mean speed for effect → cause
    forward_curvature   : mean curvature for cause → effect
    backward_curvature  : mean curvature for effect → cause
    causal_direction    : manifold causal_direction() score (positive = P causes Q)
    """

    causal_score: float
    forward_steps: int
    backward_steps: int
    forward_speed: float
    backward_speed: float
    forward_curvature: float
    backward_curvature: float
    causal_direction: float

    @classmethod
    def from_trajectories(
        cls,
        manifold: LivingManifold,
        cause_pos: np.ndarray,
        effect_pos: np.ndarray,
        forward_traj: Trajectory,
        backward_traj: Trajectory,
    ) -> "CausalMetrics":
        """Compute causal metrics from paired forward/backward trajectories."""
        # A shorter, faster forward trajectory → stronger causal score
        fwd_steps = len(forward_traj)
        bwd_steps = len(backward_traj)
        fwd_speed = float(forward_traj.mean_speed)
        bwd_speed = float(backward_traj.mean_speed)

        # Speed advantage: forward should be faster in the causal fiber
        speed_ratio = fwd_speed / (bwd_speed + 1e-10)
        # Step advantage: fewer steps forward is more directed
        # (fewer steps = less diffusion = cleaner causal path)
        step_ratio = bwd_steps / (fwd_steps + 1e-10)

        # Raw score: geometric mean of the two advantage signals, clipped to [0,1]
        raw = float(np.sqrt(speed_ratio * step_ratio))
        causal_score = float(np.clip(raw / (raw + 1.0), 0.0, 1.0))

        causal_dir_vec = manifold.causal_direction(cause_pos, effect_pos)
        causal_dir = float(np.linalg.norm(causal_dir_vec))

        return cls(
            causal_score=causal_score,
            forward_steps=fwd_steps,
            backward_steps=bwd_steps,
            forward_speed=fwd_speed,
            backward_speed=bwd_speed,
            forward_curvature=float(forward_traj.mean_curvature),
            backward_curvature=float(backward_traj.mean_curvature),
            causal_direction=causal_dir,
        )


# ── LocalityMetrics ───────────────────────────────────────────────────────────

@dataclass
class LocalityMetrics:
    """Verifies the hard LOCAL-UPDATES-ONLY constraint.

    After applying a deformation at point P, points near P may move;
    points far from P must not move (within floating-point tolerance).

    Attributes
    ----------
    locality_satisfied      : True if no distant point moved significantly
    n_nearby_moved          : count of nearby points that shifted (expected > 0)
    n_distant_moved         : count of distant points that shifted (expected = 0)
    max_nearby_shift        : maximum L2 displacement among nearby points
    max_distant_shift       : maximum L2 displacement among distant points (want ≈ 0)
    locality_radius_used    : the Gaussian falloff radius that was applied
    """

    locality_satisfied: bool
    n_nearby_moved: int
    n_distant_moved: int
    max_nearby_shift: float
    max_distant_shift: float
    locality_radius_used: float

    DISTANT_SHIFT_TOLERANCE: float = 1e-6  # absolute tolerance

    @classmethod
    def measure(
        cls,
        manifold: LivingManifold,
        anchor_label: str,
        snapshots_before: List[Tuple[str, np.ndarray]],
        snapshots_after: List[Tuple[str, np.ndarray]],
        locality_radius: float,
    ) -> "LocalityMetrics":
        """Compute locality metrics by diffing before/after position snapshots.

        Parameters
        ----------
        manifold        : the Living Manifold (used for distance queries)
        anchor_label    : the concept that was deformed
        snapshots_before: list of (label, position_vector) before deformation
        snapshots_after : list of (label, position_vector) after  deformation
        locality_radius : the radius used for the Gaussian kernel
        """
        anchor_pos = manifold.position(anchor_label)

        nearby_shifts: List[float] = []
        distant_shifts: List[float] = []

        before_dict = dict(snapshots_before)
        after_dict  = dict(snapshots_after)

        for lbl, pos_before in before_dict.items():
            if lbl not in after_dict:
                continue
            pos_after = after_dict[lbl]
            shift = float(np.linalg.norm(pos_after - pos_before))
            dist_to_anchor = float(np.linalg.norm(pos_before - anchor_pos))

            if dist_to_anchor <= 3.0 * locality_radius:
                nearby_shifts.append(shift)
            else:
                distant_shifts.append(shift)

        max_nearby  = max(nearby_shifts,  default=0.0)
        max_distant = max(distant_shifts, default=0.0)
        n_nearby_moved  = sum(s > cls.DISTANT_SHIFT_TOLERANCE for s in nearby_shifts)
        n_distant_moved = sum(s > cls.DISTANT_SHIFT_TOLERANCE for s in distant_shifts)
        locality_ok = (max_distant <= cls.DISTANT_SHIFT_TOLERANCE)

        return cls(
            locality_satisfied=locality_ok,
            n_nearby_moved=n_nearby_moved,
            n_distant_moved=n_distant_moved,
            max_nearby_shift=max_nearby,
            max_distant_shift=max_distant,
            locality_radius_used=locality_radius,
        )


# ── EvaluationResult ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Combined evaluation result for a single pipeline query.

    Attributes
    ----------
    label           : human-readable identifier for this evaluation
    coherence       : CoherenceMetrics for the query
    extra           : optional dict of task-specific scalar metrics
    """

    label: str
    coherence: CoherenceMetrics
    extra: dict = field(default_factory=dict)

    def overall_score(self) -> float:
        """Delegate to CoherenceMetrics.overall_score()."""
        return self.coherence.overall_score()

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(label={self.label!r}, "
            f"overall={self.overall_score():.3f}, "
            f"n_steps={self.coherence.trajectory_steps}, "
            f"reason={self.coherence.termination_reason!r})"
        )
