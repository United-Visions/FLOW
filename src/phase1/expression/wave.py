"""
Standing Wave — Ψ
=================
The pre-linguistic meaning representation produced by the Resonance Layer
(Component 6) and consumed by the Expression Renderer (Component 7).

Architecture relationship
-------------------------
Flow Engine (Component 5) produces a trajectory T.
Resonance Layer (Component 6) accumulates T into a standing wave Ψ:

    Ψ(Q) = ∫₀᷊ excitation(Q, t) · harmonic_factor(Q, P(t)) dt

    High Ψ = strongly resonant with the trajectory = central to meaning
    Low Ψ  = weakly resonant = peripheral
    Zero Ψ = not part of this response at all

The standing wave Ψ is a scalar field over the manifold.  In this Phase 1b
prototype we represent it as a discrete set of (concept_vector, amplitude)
pairs.  The full continuous field belongs to Phase 4.

What the Expression Renderer sees
----------------------------------
The renderer receives ONLY Ψ — it has no access to the manifold,
the trajectory, or any component above it.  Ψ is the complete interface.

Ψ encodes
---------
- Core concepts     : high amplitude positions
- Context & nuance  : medium amplitude positions
- Confidence        : clustering of high-amp regions (dense = confident)
- Uncertainty       : sparse high-amp regions in flexible zones
- Causal flow       : amplitude gradient follows causal direction
- Irrelevance       : zero amplitude

Wave segments
-------------
A WaveSegment represents one "chunk of meaning" — a connected sub-region
of the wave between two local amplitude minima.  Segmentation is done by
the Expression Renderer (Stage 1) before rendering begins.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Dimensionality of concept vectors in the wave
# Must match DIM_TOTAL from the seed geometry when connected to the full system.
# For the Phase 1b prototype we parameterise it.
WAVE_DIM = 104   # matches the seed manifold dimension

# ── Data types ─────────────────────────────────────────────────────────────


@dataclass
class WavePoint:
    """
    A single activated point in the standing wave.

    Attributes
    ----------
    vector    : concept position in 104D bundle space
    amplitude : Ψ(P) — how central this concept is to the meaning
    label     : optional human-readable label for debugging
    tau       : optional causal time coordinate (for ordering)
    """
    vector:    np.ndarray
    amplitude: float
    label:     str  = ""
    tau:       float = 0.5   # causal time; 0 = earliest cause, 1 = latest effect

    def __post_init__(self) -> None:
        self.amplitude = float(max(self.amplitude, 0.0))

    def __repr__(self) -> str:
        return f"WavePoint(label='{self.label}', amp={self.amplitude:.3f}, tau={self.tau:.2f})"


@dataclass
class WaveSegment:
    """
    A coherent chunk of meaning — a continuous region of the standing wave.

    Segments are created by the Expression Renderer Stage 1 (Segmentation).
    Each segment will be matched to a single linguistic expression.

    Attributes
    ----------
    points          : WavePoints in this segment (sorted by τ)
    mean_amplitude  : average amplitude across the segment
    peak_point      : the WavePoint with highest amplitude
    coherence       : how tightly clustered the segment is (0=diffuse, 1=tight)
    uncertainty     : how sparse/flexible the segment's manifold region is
    flow_speed      : approximate velocity when the flow passed through here
    index           : position of this segment in the ordered sequence
    """
    points:         List[WavePoint]
    mean_amplitude: float
    peak_point:     WavePoint
    coherence:      float = 0.5
    uncertainty:    float = 0.5
    flow_speed:     float = 0.5
    index:          int   = 0

    def __repr__(self) -> str:
        labels = [p.label for p in self.points if p.label][:3]
        return (
            f"WaveSegment(i={self.index}, n={len(self.points)}, "
            f"mean_amp={self.mean_amplitude:.3f}, "
            f"coherence={self.coherence:.2f}, "
            f"concepts={labels})"
        )


@dataclass
class StandingWave:
    """
    The complete standing wave Ψ — pre-linguistic meaning representation.

    This is the full output of the Resonance Layer and the complete
    input to the Expression Renderer.

    Attributes
    ----------
    points        : all activated WavePoints, ordered by decreasing amplitude
    total_energy  : ∫Ψ dM — total "meaning mass"
    query_echo    : weak representation of the original query in wave form
    n_dim         : dimensionality of concept vectors
    metadata      : free-form diagnostic information
    """
    points:       List[WavePoint]
    total_energy: float
    query_echo:   Optional[WavePoint] = None
    n_dim:        int                 = WAVE_DIM
    metadata:     Dict[str, object]  = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep sorted by descending amplitude
        self.points = sorted(self.points, key=lambda p: p.amplitude, reverse=True)
        if self.total_energy <= 0.0:
            self.total_energy = sum(p.amplitude for p in self.points) + 1e-12

    @property
    def peak(self) -> Optional[WavePoint]:
        """The highest-amplitude point — core of the meaning."""
        return self.points[0] if self.points else None

    @property
    def normalised_amplitudes(self) -> np.ndarray:
        """All amplitudes normalised to [0, 1] by dividing by the peak."""
        if not self.points:
            return np.array([])
        amps = np.array([p.amplitude for p in self.points])
        return amps / (amps.max() + 1e-12)

    def top_k(self, k: int) -> List[WavePoint]:
        """Return the k most activated points."""
        return self.points[:k]

    def above_threshold(self, threshold: float = 0.1) -> List[WavePoint]:
        """Return all points with normalised amplitude ≥ threshold."""
        norm = self.normalised_amplitudes
        return [p for p, a in zip(self.points, norm) if a >= threshold]

    def confident_core(self, threshold: float = 0.4) -> List[WavePoint]:
        """Return the high-amplitude core of the wave (the certain meaning)."""
        return self.above_threshold(threshold)

    def peripheral(self, lo: float = 0.05, hi: float = 0.4) -> List[WavePoint]:
        """Return supporting/contextual points (medium amplitude)."""
        norm = self.normalised_amplitudes
        return [p for p, a in zip(self.points, norm) if lo <= a < hi]

    def mean_confidence(self) -> float:
        """Overall confidence as a function of wave concentration."""
        if not self.points:
            return 0.0
        amps = np.array([p.amplitude for p in self.points])
        amps /= (amps.sum() + 1e-12)
        # Entropy-based confidence: 0 = diffuse, 1 = concentrated
        entropy = -np.sum(amps * np.log(amps + 1e-12))
        max_entropy = np.log(len(amps) + 1e-12)
        return float(1.0 - entropy / max_entropy)

    def mean_uncertainty(self) -> float:
        return 1.0 - self.mean_confidence()

    def __repr__(self) -> str:
        return (
            f"StandingWave(n_points={len(self.points)}, "
            f"energy={self.total_energy:.3f}, "
            f"confidence={self.mean_confidence():.2f})"
        )


# ── Mock wave constructors ─────────────────────────────────────────────────


def create_mock_wave(
    theme: str,
    n_core: int = 8,
    n_support: int = 16,
    seed: int = 0,
) -> StandingWave:
    """
    Create a hand-crafted standing wave for testing the Expression Renderer.

    The mock wave encodes a semantic theme as a structured Ψ field without
    requiring the full manifold pipeline.

    Parameters
    ----------
    theme   : str
        The semantic theme.  Accepted values (with fallback to generic):
        'explanation', 'causation', 'uncertainty', 'contrast', 'discovery',
        'warning', 'instruction', 'conclusion'.
    n_core  : int
        Number of high-amplitude core points (the main message).
    n_support : int
        Number of medium-amplitude support points (context/nuance).
    seed    : int
        Random seed for reproducibility.

    Returns
    -------
    StandingWave
        A mock wave with structured amplitude distribution.
    """
    rng = np.random.default_rng(seed)

    # Theme configs — each theme has a characteristic wave shape
    theme_configs = {
        "explanation": {
            "tau_range":    (0.0, 1.0),   # full causal chain
            "coherence":    0.7,
            "uncertainty":  0.2,
            "concepts":     ["cause", "mechanism", "effect", "reason", "because",
                             "therefore", "results_in", "leads_to"],
            "flow_speed":   0.5,
        },
        "causation": {
            "tau_range":    (0.2, 0.9),
            "coherence":    0.8,
            "uncertainty":  0.15,
            "concepts":     ["trigger", "force", "propagation", "consequence",
                             "chain", "direct_effect", "mechanism"],
            "flow_speed":   0.7,
        },
        "uncertainty": {
            "tau_range":    (0.3, 0.7),
            "coherence":    0.3,
            "uncertainty":  0.8,
            "concepts":     ["possibility", "likely", "perhaps", "estimate",
                             "approximate", "unclear", "range"],
            "flow_speed":   0.3,
        },
        "contrast": {
            "tau_range":    (0.1, 0.8),
            "coherence":    0.6,
            "uncertainty":  0.3,
            "concepts":     ["difference", "distinction", "whereas", "however",
                             "alternatively", "unlike", "comparison"],
            "flow_speed":   0.6,
        },
        "discovery": {
            "tau_range":    (0.0, 0.6),
            "coherence":    0.5,
            "uncertainty":  0.5,
            "concepts":     ["observation", "finding", "evidence", "reveals",
                             "indicates", "suggests", "pattern"],
            "flow_speed":   0.4,
        },
        "warning": {
            "tau_range":    (0.5, 1.0),
            "coherence":    0.9,
            "uncertainty":  0.1,
            "concepts":     ["risk", "danger", "consequence", "avoid", "caution",
                             "prevent", "attention"],
            "flow_speed":   0.9,
        },
        "instruction": {
            "tau_range":    (0.0, 1.0),
            "coherence":    0.85,
            "uncertainty":  0.1,
            "concepts":     ["step", "action", "sequence", "first", "then",
                             "next", "finally", "procedure"],
            "flow_speed":   0.65,
        },
        "conclusion": {
            "tau_range":    (0.6, 1.0),
            "coherence":    0.8,
            "uncertainty":  0.2,
            "concepts":     ["therefore", "overall", "summary", "thus",
                             "demonstrates", "confirms", "result"],
            "flow_speed":   0.5,
        },
    }
    cfg = theme_configs.get(theme, theme_configs["explanation"])
    tau_lo, tau_hi = cfg["tau_range"]
    concept_labels  = cfg["concepts"]
    coherence       = cfg["coherence"]
    uncertainty_val = cfg["uncertainty"]
    flow_speed      = cfg["flow_speed"]

    points: List[WavePoint] = []

    # ── Core high-amplitude points ─────────────────────────────────────
    # Core vectors are clustered (coherent meaning)
    core_centre = rng.uniform(-0.3, 0.3, WAVE_DIM)
    core_centre /= (np.linalg.norm(core_centre) + 1e-12)

    for i in range(n_core):
        spread    = 0.2 * (1.0 - coherence)
        vec       = core_centre + rng.normal(0, spread, WAVE_DIM)
        amplitude = rng.uniform(0.6, 1.0)   # high amplitude
        tau       = tau_lo + (tau_hi - tau_lo) * i / max(n_core - 1, 1)
        label     = concept_labels[i % len(concept_labels)]
        points.append(WavePoint(vector=vec, amplitude=amplitude, label=label, tau=tau))

    # ── Support medium-amplitude points ────────────────────────────────
    for j in range(n_support):
        spread    = 0.5 * (1.0 + uncertainty_val)
        vec       = core_centre + rng.normal(0, spread, WAVE_DIM)
        amplitude = rng.uniform(0.1, 0.55)  # medium amplitude
        tau       = rng.uniform(tau_lo, tau_hi)
        points.append(WavePoint(vector=vec, amplitude=amplitude, label="", tau=tau))

    total_energy = sum(p.amplitude for p in points)

    # Query echo: a weak representation of the query
    query_vec = core_centre + rng.normal(0, 0.3, WAVE_DIM)
    query_echo = WavePoint(
        vector=query_vec,
        amplitude=0.05,
        label=f"query::{theme}",
        tau=0.0,
    )

    return StandingWave(
        points=points,
        total_energy=total_energy,
        query_echo=query_echo,
        n_dim=WAVE_DIM,
        metadata={
            "theme":      theme,
            "coherence":  coherence,
            "uncertainty": uncertainty_val,
            "flow_speed": flow_speed,
            "n_core":     n_core,
            "n_support":  n_support,
        },
    )


def create_wave_from_trajectory(
    trajectory: List[Tuple[np.ndarray, float]],
    resonance_radius: float = 0.5,
    harmonic_decay: float   = 0.7,
) -> StandingWave:
    """
    Build a standing wave from a trajectory (list of (position, time) pairs).

    This is the real Resonance Layer (Phase 4) simplified for testing.
    Implements the excitation–accumulation logic from the architecture spec:

        Ψ(Q) = Σₜ A · exp(- ‖Q - P(t)‖² / 2r² ) · harmonic_factor(Q, t)

    Parameters
    ----------
    trajectory         : list of (position_vector, time) tuples
    resonance_radius   : the r in the Gaussian excitation kernel
    harmonic_decay     : controls how quickly harmonic amplification falls off

    Returns
    -------
    StandingWave
    """
    if not trajectory:
        return StandingWave(points=[], total_energy=0.0)

    # Use trajectory points themselves as Ψ activation sites
    positions = np.array([pos for pos, _ in trajectory])
    times     = np.array([t   for _, t   in trajectory])

    # Normalise times to [0, 1]
    t_max = times.max() + 1e-12
    times /= t_max

    # Compute amplitude at each trajectory point via resonance accumulation
    n = len(trajectory)
    amplitudes = np.zeros(n)

    for i in range(n):
        # Self-excitation: Ψ grows with each step
        amplitudes[i] += 1.0

        # Harmonic amplification from nearby earlier positions
        for j in range(i):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            dt   = times[i] - times[j]
            if dt > 0:
                excitation = np.exp(-dist ** 2 / (2 * resonance_radius ** 2))
                harmonic   = harmonic_decay ** (dt * 10)
                amplitudes[i] += excitation * harmonic

    points = [
        WavePoint(
            vector=positions[i],
            amplitude=float(amplitudes[i]),
            label=f"t_{i}",
            tau=float(times[i]),
        )
        for i in range(n)
    ]

    return StandingWave(
        points=points,
        total_energy=float(amplitudes.sum()),
        n_dim=positions.shape[1] if len(positions) > 0 else WAVE_DIM,
    )
