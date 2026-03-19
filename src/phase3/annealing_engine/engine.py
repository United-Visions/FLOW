"""Annealing Engine — continuous self-organisation of the Living Manifold.

Component 3 of the Geometric Causal Architecture.

Responsibility:
  Shape the manifold from raw unlabeled experience using physics-inspired
  self-organisation.  No labels.  No supervision.  No gradients.

What it is NOT responsible for:
  Creating the seed geometry (C1).
  Placing specific named concepts (C4 Contrast Engine).
  Generating output (C7 Expression Renderer).

Experience-processing loop (per experience E):

  1. LOCATE     — find the natural resonance position P via kNN
  2. NOVELTY    — score how surprising E is: f(distance, density)
  3. DEFORM     — δ = novelty · T(t) · consistency_gradient
  4. APPLY      — M.deform_local(P, δ)   [locality guaranteed]
  5. DENSITY    — M.update_density(P)

Temperature schedule:
  T(t) = T₀ · e^(-λt) + T_floor

  High T → wide, exploratory deformations (coarse structure)
  Low  T → narrow, precise deformations  (fine-tuning)
  T_floor → unknown territory always stays plastic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.phase2.living_manifold.manifold import LivingManifold
from .experience import Experience, ExperienceResult
from .novelty import NoveltyEstimator, NoveltyResult
from .schedule import TemperatureSchedule


# ─────────────────────────────────────────────────────────────────────────────
# AnnealingStats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnealingStats:
    """Running statistics tracked by the AnnealingEngine.

    Attributes
    ----------
    n_processed       : total experiences processed
    n_novel           : experiences with novelty > 0.5
    total_deformation : cumulative sum of |δ| across all experiences
    mean_novelty      : rolling mean novelty score
    mean_temperature  : rolling mean temperature at processing time
    """

    n_processed: int = 0
    n_novel: int = 0
    total_deformation: float = 0.0
    _novelty_sum: float = 0.0
    _temperature_sum: float = 0.0

    def record(self, result: ExperienceResult) -> None:
        """Update running stats with a new result."""
        self.n_processed += 1
        if result.was_novel:
            self.n_novel += 1
        self.total_deformation += result.delta_magnitude
        self._novelty_sum += result.novelty
        self._temperature_sum += result.temperature

    @property
    def mean_novelty(self) -> float:
        if self.n_processed == 0:
            return 0.0
        return self._novelty_sum / self.n_processed

    @property
    def mean_temperature(self) -> float:
        if self.n_processed == 0:
            return 0.0
        return self._temperature_sum / self.n_processed

    @property
    def novelty_rate(self) -> float:
        if self.n_processed == 0:
            return 0.0
        return self.n_novel / self.n_processed


# ─────────────────────────────────────────────────────────────────────────────
# AnnealingEngine
# ─────────────────────────────────────────────────────────────────────────────

class AnnealingEngine:
    """Continuously self-organise the Living Manifold from raw experience.

    Parameters
    ----------
    manifold : LivingManifold
        The living manifold to deform.
    T0 : float
        Initial annealing temperature.  Default 1.0.
    lambda_ : float
        Cooling rate.  Default 0.01.
    T_floor : float
        Minimum temperature; system never fully freezes.  Default 0.05.
    base_radius : float
        Maximum deformation radius at T=T_max.  Passed to the temperature
        schedule to derive the temperature-scaled locality radius.
        Default 5.0.
    k_neighbors : int
        Number of nearest neighbours used for resonance location and the
        consistency gradient.  Default 5.
    place_labeled : bool
        If True and an Experience carries a label, the experience vector
        is placed as a named concept on the manifold.  Default True.
    novelty_threshold : float
        Minimum novelty required to actually apply a deformation.
        Experiences below this threshold are located but not applied.
        Default 0.01.
    """

    def __init__(
        self,
        manifold: LivingManifold,
        T0: float = 1.0,
        lambda_: float = 0.01,
        T_floor: float = 0.05,
        base_radius: float = 5.0,
        k_neighbors: int = 5,
        place_labeled: bool = True,
        novelty_threshold: float = 0.01,
    ) -> None:
        self._manifold = manifold
        self._schedule = TemperatureSchedule(
            T0=T0, lambda_=lambda_, T_floor=T_floor
        )
        self._novelty = NoveltyEstimator()
        self._base_radius = base_radius
        self._k_neighbors = k_neighbors
        self._place_labeled = place_labeled
        self._novelty_threshold = novelty_threshold
        self._stats = AnnealingStats()
        self._history: List[ExperienceResult] = []

    # ------------------------------------------------------------------ #
    # Core processing                                                      #
    # ------------------------------------------------------------------ #

    def process(self, experience: Experience) -> ExperienceResult:
        """Process a single raw experience and deform the manifold.

        Steps:
          1. LOCATE — resonance: find nearest existing point
          2. NOVELTY — score against local neighbourhood
          3. DEFORM — scale consistency gradient by novelty × T(t)
          4. APPLY — M.deform_local(anchor, δ)
          5. DENSITY — M.update_density(anchor)

        Parameters
        ----------
        experience : Experience
            Raw experience to ingest.

        Returns
        -------
        ExperienceResult with full processing record.
        """
        vec = experience.vector

        # ── Step 1: LOCATE via resonance (kNN) ──────────────────────────
        anchor_label, anchor_pos = self._locate(vec)

        # ── Step 2: NOVELTY ─────────────────────────────────────────────
        neighbor_positions = self._get_neighbor_positions(anchor_pos)
        local_density = self._manifold.density(anchor_pos)
        novelty_result = self._novelty.estimate(
            vec, neighbor_positions, local_density
        )

        # ── Step 3: DEFORMATION VECTOR ───────────────────────────────────
        T_current = self._schedule.current_temperature
        gradient = self._novelty.consistency_gradient(vec, neighbor_positions)

        delta = novelty_result.score * T_current * gradient

        # ── Step 4: APPLY (via manifold WRITE) ───────────────────────────
        placed_label: Optional[str] = None
        n_affected = 0

        if anchor_label is not None and novelty_result.score >= self._novelty_threshold:
            n_affected = self._manifold.deform_local(anchor_label, delta)

        # Optionally place labeled experiences on the manifold
        if self._place_labeled and experience.label is not None:
            mp = self._manifold.place(experience.label, vec)
            placed_label = mp.label
            # ── Step 5: DENSITY ─────────────────────────────────────────
            self._manifold.update_density(experience.label)
        elif anchor_label is not None:
            # ── Step 5: DENSITY for anchor ───────────────────────────────
            self._manifold.update_density(anchor_label)

        # Tick the temperature schedule
        self._schedule.step()

        result = ExperienceResult(
            experience=experience,
            located_label=anchor_label,
            located_position=anchor_pos if anchor_pos is not None else vec.copy(),
            novelty=novelty_result.score,
            temperature=T_current,
            delta_magnitude=float(np.linalg.norm(delta)),
            n_affected=n_affected,
            placed_label=placed_label,
        )

        self._stats.record(result)
        self._history.append(result)

        return result

    def process_batch(
        self, experiences: List[Experience]
    ) -> List[ExperienceResult]:
        """Process a list of experiences in sequence.

        Returns results in the same order as the input list.
        """
        return [self.process(e) for e in experiences]

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    @property
    def temperature(self) -> float:
        """Current temperature of the schedule."""
        return self._schedule.current_temperature

    @property
    def t(self) -> float:
        """Current time of the temperature schedule."""
        return self._schedule.t

    @property
    def n_processed(self) -> int:
        """Total number of experiences processed so far."""
        return self._stats.n_processed

    @property
    def stats(self) -> AnnealingStats:
        """Running statistics object."""
        return self._stats

    @property
    def schedule(self) -> TemperatureSchedule:
        """Access the underlying temperature schedule."""
        return self._schedule

    def reset_temperature(self) -> None:
        """Reset the temperature schedule back to T₀.

        Does NOT undo any manifold deformations.
        """
        self._schedule.reset()

    def summary(self) -> str:
        """Return a human-readable summary string."""
        s = self._stats
        return (
            f"AnnealingEngine:\n"
            f"  processed        : {s.n_processed}\n"
            f"  novel (>0.5)     : {s.n_novel}  ({s.novelty_rate:.1%})\n"
            f"  mean novelty     : {s.mean_novelty:.3f}\n"
            f"  total deformation: {s.total_deformation:.4f}\n"
            f"  temperature now  : {self.temperature:.4f}\n"
            f"  schedule time    : {self.t:.1f}\n"
            f"  manifold writes  : {self._manifold.n_writes}\n"
            f"  manifold points  : {self._manifold.n_points}"
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _locate(
        self, vec: np.ndarray
    ) -> Tuple[Optional[str], np.ndarray]:
        """Find the resonance anchor for *vec* via kNN search.

        Returns (label, position) of the single nearest labelled point.
        If the manifold is empty, returns (None, vec).
        """
        nearest = self._manifold.nearest(vec, k=1)
        if not nearest:
            return None, vec.copy()
        label, _ = nearest[0]
        pos = self._manifold.position(label)
        return label, pos

    def _get_neighbor_positions(
        self, anchor_pos: np.ndarray
    ) -> List[np.ndarray]:
        """Return position vectors for up to k_neighbors near *anchor_pos*."""
        nearest = self._manifold.nearest(anchor_pos, k=self._k_neighbors)
        return [self._manifold.position(lbl) for lbl, _ in nearest]
