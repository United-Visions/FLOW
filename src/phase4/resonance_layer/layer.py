"""Resonance Layer — Component 6 of the Geometric Causal Architecture.

Responsibility:
  Convert the flow trajectory T into a holistic pre-linguistic meaning
  representation — a standing wave Ψ across the manifold.  The entire
  response exists here before rendering begins.

What it is NOT responsible for:
  Navigation (C5).
  Language production (C7).
  Storing knowledge (C2).

Process:
  For each step P(t) in the trajectory:
    1. EXCITE  all sites Q near P: excitation(Q,t) = A·exp(−‖Q−P‖²/2r²)
    2. AMPLIFY harmonically related sites
    3. ACCUMULATE Ψ(Q) += excitation(Q,t) · harmonic_factor(Q, P(t))

  Output: StandingWave Ψ — scalar field over activated manifold points.
  High Ψ  = core meaning.
  Low  Ψ  = peripheral context.
  Zero Ψ  = not part of this response.

Interface to C7:
  The ResonanceLayer produces a StandingWave (defined in phase1/expression/wave.py).
  This is the ONLY data C7 receives — it has no manifold access at all.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.phase1.expression.wave import StandingWave, WavePoint, WAVE_DIM
from .accumulator import ResonanceAccumulator
from ..flow_engine.query import Trajectory


# ─────────────────────────────────────────────────────────────────────────────
# ResonanceLayer
# ─────────────────────────────────────────────────────────────────────────────

class ResonanceLayer:
    """Convert a flow trajectory into a standing wave Ψ.

    Parameters
    ----------
    manifold : LivingManifold
        Read-only manifold access for curvature and density queries.
    resonance_radius : float
        Base Gaussian excitation radius.  Default 0.5.
    harmonic_tolerance : float
        Half-width of harmonic matching kernel.  Default 0.15.
    amplitude_floor : float
        Minimum normalised amplitude for a WavePoint to be included in Ψ.
        Points below this are considered silent.  Default 0.01.
    """

    def __init__(
        self,
        manifold,
        resonance_radius: float = 0.5,
        harmonic_tolerance: float = 0.15,
        amplitude_floor: float = 0.01,
    ) -> None:
        if resonance_radius <= 0:
            raise ValueError(
                f"resonance_radius must be positive, got {resonance_radius}"
            )
        if amplitude_floor < 0:
            raise ValueError(
                f"amplitude_floor must be non-negative, got {amplitude_floor}"
            )
        self._manifold = manifold
        self._accumulator = ResonanceAccumulator(
            resonance_radius=resonance_radius,
            harmonic_tolerance=harmonic_tolerance,
        )
        self.amplitude_floor = amplitude_floor

    # ── Public interface ─────────────────────────────────────────────── #

    def accumulate(self, trajectory: Trajectory) -> StandingWave:
        """Convert *trajectory* into a standing wave Ψ.

        Parameters
        ----------
        trajectory : Trajectory
            Output of the Flow Engine.

        Returns
        -------
        StandingWave
            The pre-linguistic meaning representation ready for C7.
        """
        if trajectory.is_empty:
            return StandingWave(
                points=[],
                total_energy=0.0,
                query_echo=self._build_query_echo(trajectory),
                n_dim=WAVE_DIM,
                metadata={"n_trajectory_steps": 0},
            )

        # ── Build excitation sites from trajectory steps ────────────── #
        sites = self._build_sites(trajectory)

        # ── Accumulate Ψ at each site ───────────────────────────────── #
        raw_amplitudes = self._accumulator.accumulate(sites)

        # ── Normalise amplitudes to [0, 1] ──────────────────────────── #
        peak = float(raw_amplitudes.max()) if len(raw_amplitudes) > 0 else 1.0
        if peak < 1e-12:
            peak = 1.0
        norm_amplitudes = raw_amplitudes / peak

        # ── Build WavePoints ────────────────────────────────────────── #
        steps = trajectory.steps
        n_total = len(steps)
        t_max = steps[-1].time if steps else 1.0

        wave_points: List[WavePoint] = []
        _recent_labels: List[str] = []   # sliding-window dedup (last 3 labels)
        for i, (step, amp_norm) in enumerate(zip(steps, norm_amplitudes)):
            if amp_norm < self.amplitude_floor:
                continue
            tau = step.time / max(t_max, 1e-12)
            # Resolve the nearest manifold concept so C7 gets meaningful labels
            # instead of positional placeholders like "flow_t{i}".
            # Use a sliding window so a tight revisiting trajectory draws labels
            # from the top-5 nearby concepts rather than repeating one forever.
            try:
                candidates = self._manifold.nearest(step.position, k=5)
                label = f"flow_t{i}"
                for cand_label, _ in candidates:
                    if cand_label not in _recent_labels[-3:]:
                        label = cand_label
                        break
                else:
                    # All top-5 recently used — fall back to the closest
                    label = candidates[0][0] if candidates else f"flow_t{i}"
            except Exception:
                label = f"flow_t{i}"
            _recent_labels.append(label)
            wave_points.append(
                WavePoint(
                    vector=step.position.copy(),
                    amplitude=float(amp_norm),
                    label=label,
                    tau=float(tau),
                )
            )

        # ── Query echo: weak representation of the original query ────── #
        query_echo = self._build_query_echo(trajectory)

        total_energy = float(sum(p.amplitude for p in wave_points)) + 1e-12

        return StandingWave(
            points=wave_points,
            total_energy=total_energy,
            query_echo=query_echo,
            n_dim=WAVE_DIM,
            metadata={
                "n_trajectory_steps": n_total,
                "n_wave_points": len(wave_points),
                "termination_reason": trajectory.termination_reason,
                "trajectory_mean_speed": trajectory.mean_speed,
                "trajectory_mean_curvature": trajectory.mean_curvature,
            },
        )

    # ── Internal helpers ─────────────────────────────────────────────── #

    def _build_sites(
        self, trajectory: Trajectory
    ) -> List[Tuple[np.ndarray, float, float]]:
        """Collect (position, speed, curvature) for each trajectory step."""
        sites = []
        for step in trajectory.steps:
            kappa = self._manifold.curvature(step.position)
            sites.append((step.position, step.speed, kappa))
        return sites

    def _build_query_echo(self, trajectory: Trajectory) -> Optional[WavePoint]:
        """Weak wave representation of the originating query."""
        q_vec = trajectory.query.vector
        return WavePoint(
            vector=q_vec.copy(),
            amplitude=0.05,
            label=f"query::{trajectory.query.label}" if trajectory.query.label else "query",
            tau=0.0,
        )
