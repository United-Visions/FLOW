"""Excitation and harmonic-amplification kernels for the Resonance Layer.

Architecture spec (Component 6):

  1. EXCITATION
     excitation(Q, t) = A · exp(−‖Q − P(t)‖² / (2r²))
     A = flow speed at step t   (faster flow = stronger excitation)
     r = resonance_radius scaled by local curvature
         (more curved regions resonate narrowly)

  2. HARMONIC AMPLIFICATION
     Two positions are harmonically related when their characteristic
     frequencies are (approximately) integer multiples of each other.

     freq(P) = curvature(P) + 1.0   (shift away from zero)

     ratio   = freq(P) / freq(Q)  (or its reciprocal if < 1)
     nearest_integer_distance = |ratio − round(ratio)|
     harmonic_factor = exp(−nearest_integer_distance² / (2 · τ²))
       τ = harmonic_tolerance (smaller = stricter integer matching)

  3. ACCUMULATION
     Ψ(Q) = Σₜ  excitation(Q, t) · harmonic_factor(Q, P(t))
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


class ExcitationKernel:
    """Compute the Gaussian excitation emitted by one flow step.

    Parameters
    ----------
    resonance_radius : float
        Base radius r for the excitation Gaussian.  Adjusted by curvature.
    min_radius : float
        Minimum clamped radius (prevents singularity in high-curvature regions).
    """

    def __init__(
        self,
        resonance_radius: float = 0.5,
        min_radius: float = 0.05,
    ) -> None:
        if resonance_radius <= 0:
            raise ValueError(f"resonance_radius must be positive, got {resonance_radius}")
        self.resonance_radius = resonance_radius
        self.min_radius = min_radius

    def effective_radius(self, curvature: float) -> float:
        """r_eff = resonance_radius / (1 + curvature).

        High curvature → narrow resonance cone; low curvature → wide.
        """
        r = self.resonance_radius / (1.0 + max(curvature, 0.0))
        return max(r, self.min_radius)

    def excitation(
        self,
        query_pos: np.ndarray,
        flow_pos: np.ndarray,
        speed: float,
        curvature: float,
    ) -> float:
        """A · exp(−‖Q − P‖² / (2 r_eff²)).

        A = flow speed (clamped to [0.01, ∞) to avoid zero amplitude even
        when the flow stalls momentarily).
        """
        r = self.effective_radius(curvature)
        dist_sq = float(np.dot(query_pos - flow_pos, query_pos - flow_pos))
        amplitude = max(speed, 0.01)
        return float(amplitude * math.exp(-dist_sq / (2.0 * r * r)))


class HarmonicKernel:
    """Compute the harmonic amplification factor between two manifold points.

    The characteristic frequency of a point P is derived from its local
    curvature (a property of the manifold geometry):

        freq(P) = curvature(P) + 1.0

    Two points are harmonically related when freq_ratio ≈ integer.

    Parameters
    ----------
    harmonic_tolerance : float
        Half-width τ of the harmonic Gaussian.  Smaller = stricter.
        Default 0.15  (roughly "close to an integer" to within 15% of a
        half-cycle).
    """

    def __init__(self, harmonic_tolerance: float = 0.15) -> None:
        if harmonic_tolerance <= 0:
            raise ValueError(
                f"harmonic_tolerance must be positive, got {harmonic_tolerance}"
            )
        self.harmonic_tolerance = harmonic_tolerance

    def factor(self, curvature_q: float, curvature_p: float) -> float:
        """Harmonic factor ∈ [0, 1] between positions with given curvatures.

        Returns 1.0 when freq_q / freq_p ≈ integer, < 1 otherwise.
        """
        freq_q = curvature_q + 1.0
        freq_p = curvature_p + 1.0
        ratio = freq_q / freq_p
        # Distance to nearest integer
        nearest_int = round(ratio)
        if nearest_int < 1:
            nearest_int = 1
        delta = abs(ratio - nearest_int)
        tau = self.harmonic_tolerance
        return float(math.exp(-delta ** 2 / (2.0 * tau * tau)))


class ResonanceAccumulator:
    """Accumulate excitations from a trajectory into a standing wave.

    Ψ(Q) = Σₜ  excitation(Q, t) · harmonic_factor(Q, P(t))

    Usage
    -----
    acc = ResonanceAccumulator()
    amplitudes = acc.accumulate(positions_and_steps)

    Parameters
    ----------
    resonance_radius : float  — passed to ExcitationKernel
    harmonic_tolerance : float — passed to HarmonicKernel
    """

    def __init__(
        self,
        resonance_radius: float = 0.5,
        harmonic_tolerance: float = 0.15,
    ) -> None:
        self._excitation = ExcitationKernel(resonance_radius=resonance_radius)
        self._harmonic = HarmonicKernel(harmonic_tolerance=harmonic_tolerance)

    def accumulate(
        self,
        sites: List[Tuple[np.ndarray, float, float]],  # (position, speed, curvature)
    ) -> np.ndarray:
        """Compute Ψ amplitude for each site.

        Each site serves simultaneously as a query point Q and as a flow
        step P(t).  The accumulation is:

            Ψ(Q_i) = Σ_j  excitation(Q_i, P_j, speed_j, κ_j)
                          · harmonic_factor(κ_i, κ_j)

        Parameters
        ----------
        sites : list of (position, speed, curvature)

        Returns
        -------
        np.ndarray of shape (n_sites,) — Ψ amplitude at each site
        """
        n = len(sites)
        if n == 0:
            return np.array([])

        amplitudes = np.zeros(n)

        for i, (q_pos, _q_speed, q_kappa) in enumerate(sites):
            psi_i = 0.0
            for j, (p_pos, p_speed, p_kappa) in enumerate(sites):
                exc = self._excitation.excitation(q_pos, p_pos, p_speed, p_kappa)
                har = self._harmonic.factor(q_kappa, p_kappa)
                psi_i += exc * har
            amplitudes[i] = psi_i

        return amplitudes
