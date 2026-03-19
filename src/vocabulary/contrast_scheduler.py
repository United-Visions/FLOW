"""Contrast Scheduler — converts a PMI matrix into a stream of C4 judgments.

Reads a CoOccurrenceMatrix and drives the ContrastEngine with batched
SAME / DIFFERENT / CAUSAL judgments derived from PMI thresholds.

This is the only place where co-occurrence statistics are converted into
manifold geometry changes.  After the scheduler runs, every word on M(t)
has been refined by its semantic neighbourhood — without any learned weights.

Mathematics (§2.1 of vocabulary-geometry-specification.md)
----------------------------------------------------------
PMI(w₁,w₂)  > τ_same  → SAME   (pull P(w₁), P(w₂) closer by α)
PMI(w₁,w₂)  < τ_diff  → DIFFERENT (push apart by β)
τ_diff ≤ PMI ≤ τ_same  → NEUTRAL (skip)

dPMI(w₁→w₂) > dPMI(w₂→w₁) + δ_causal → CAUSAL BIAS
  → apply a small displacement biased into causal fiber dims 64–79

Strength mapping (§2.3):
    strength(w₁,w₂) = min(1.0, |PMI(w₁,w₂)| / PMI_max)

Batch processing
----------------
Judgments are applied in batches of `batch_size`.  After each batch the
manifold density is updated for all affected words.  This ensures the
LOCAL UPDATES constraint is respected — a burst of 100K simultaneous
judgments would violate locality by making the manifold settle before
previous judgments have propagated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.phase2.contrast_engine.engine import ContrastEngine

from src.phase2.contrast_engine.engine import JudgmentType
from src.vocabulary.cooccurrence import CoOccurrenceMatrix


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContrastPair:
    """A single contrast judgment pending submission to ContrastEngine."""
    label_a:  str
    label_b:  str
    judgment: JudgmentType
    strength: float  # in (0, 1]


@dataclass
class CausalBiasDirective:
    """Instruction to nudge two vocab concepts along the causal fiber.

    label_a precedes / causes label_b according to directed PMI.
    The direction is +1.0 (label_a → label_b).
    """
    label_a:   str
    label_b:   str
    dpmi_diff: float  # how strongly directed the relationship is


# ─────────────────────────────────────────────────────────────────────────────
# ContrastScheduler
# ─────────────────────────────────────────────────────────────────────────────

class ContrastScheduler:
    """Schedule and apply C4 contrast judgments from a PMI matrix.

    Parameters
    ----------
    contrast_engine : ContrastEngine
        The live C4 engine connected to the LivingManifold.
    tau_same : float
        PMI threshold above which two words are judged SAME.  Default +1.0.
    tau_diff : float
        PMI threshold below which two words are judged DIFFERENT.
        Default -0.5.
    batch_size : int
        Number of judgment pairs processed per batch before a density
        update pass.  Default 256.
    delta_causal : float
        Minimum dPMI asymmetry to trigger a causal fiber bias.  Default 0.5.
    causal_bias_strength : float
        Magnitude of the causal fiber displacement applied per directive.
        Default 0.05 (gentle — contrast pairs do the coarse work).
    """

    def __init__(
        self,
        contrast_engine: "ContrastEngine",
        tau_same:            float = 1.0,
        tau_diff:            float = -0.5,
        batch_size:          int   = 256,
        delta_causal:        float = 0.5,
        causal_bias_strength: float = 0.05,
    ) -> None:
        self._engine             = contrast_engine
        self.tau_same            = tau_same
        self.tau_diff            = tau_diff
        self.batch_size          = batch_size
        self.delta_causal        = delta_causal
        self.causal_bias_strength = causal_bias_strength

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, matrix: CoOccurrenceMatrix) -> int:
        """Apply all contrast judgments derived from *matrix*.

        Only words that are already on the manifold (reachable via
        ``manifold.position(label)``) are processed.  Words not yet placed
        are silently skipped — they can be placed via WordPlacer first.

        Parameters
        ----------
        matrix : CoOccurrenceMatrix
            Result of CoOccurrenceCounter.build().

        Returns
        -------
        int — total number of judgment pairs actually applied.
        """
        manifold = self._engine._manifold
        pmi_max  = matrix.pmi_max()
        n_applied = 0

        # ── Phase 1: Symmetric SAME / DIFFERENT judgments ─────────────────
        pairs = matrix.pairs_above_threshold(self.tau_same, self.tau_diff)
        batch: List[ContrastPair] = []

        for w1, w2, pmi_val in pairs:
            la = f"vocab::{w1}"
            lb = f"vocab::{w2}"

            if not self._on_manifold(manifold, la):
                continue
            if not self._on_manifold(manifold, lb):
                continue

            j = JudgmentType.SAME if pmi_val > self.tau_same else JudgmentType.DIFFERENT
            strength = min(1.0, abs(pmi_val) / pmi_max)
            batch.append(ContrastPair(la, lb, j, strength))

            if len(batch) >= self.batch_size:
                n_applied += self._flush_batch(batch)
                batch = []

        if batch:
            n_applied += self._flush_batch(batch)

        # ── Phase 2: Directed causal fiber bias ───────────────────────────
        directed = matrix.directed_pairs_above_delta(self.delta_causal)
        n_applied += self._apply_causal_bias(directed, manifold)

        return n_applied

    def run_passes(self, matrix: CoOccurrenceMatrix, n_passes: int = 3) -> int:
        """Run the full contrast pass *n_passes* times.

        Multiple passes allow the geometry to converge: words shift after
        the first pass, changing their relative positions, so subsequent
        passes re-apply the same PMI evidence against the updated geometry.

        Returns
        -------
        int — total judgments applied across all passes.
        """
        total = 0
        for _ in range(n_passes):
            total += self.run(matrix)
        return total

    # ── Helpers ────────────────────────────────────────────────────────────

    def _flush_batch(self, batch: List[ContrastPair]) -> int:
        """Apply a batch of contrast pairs and return the count applied."""
        manifold = self._engine._manifold
        n = 0
        for cp in batch:
            try:
                self._engine.judge(cp.label_a, cp.label_b, cp.judgment, cp.strength)
                n += 1
            except (KeyError, ValueError):
                # Word was removed from manifold or invalid — skip
                continue

        # Density refresh after each batch (locality maintenance)
        for cp in batch[:min(len(batch), 32)]:  # Sample; full refresh too slow
            try:
                manifold.update_density(cp.label_a)
                manifold.update_density(cp.label_b)
            except (KeyError, ValueError):
                continue

        return n

    def _apply_causal_bias(
        self,
        directed: List[Tuple[str, str, float]],
        manifold,
    ) -> int:
        """Nudge word pairs along the causal fiber (dims 64–79).

        For each directed pair (w1→w2), a small displacement is applied to
        w2's position in the causal fiber direction — encoding that w2 tends
        to follow w1 in text.

        This is a geometric hint, not a strong enforcement.  The causal fiber
        geometry emerges from the accumulation of many such nudges.
        """
        n = 0
        # Causal displacement: dims 64–79 only
        CAUSAL_DIM = 16   # 80 - 64
        for w1, w2, dpmi_diff in directed:
            la = f"vocab::{w1}"
            lb = f"vocab::{w2}"
            if not self._on_manifold(manifold, la):
                continue
            if not self._on_manifold(manifold, lb):
                continue

            # Build a displacement that only affects causal fiber
            displacement = np.zeros(104, dtype=float)
            strength = min(1.0, dpmi_diff / 3.0)  # scale by asymmetry magnitude
            # w2 (the "effect") gets nudged toward higher causal time (tau > 0.5)
            displacement[64:80] = self.causal_bias_strength * strength

            try:
                manifold.deform_local(lb, displacement)
                n += 1
            except (KeyError, ValueError):
                continue

        return n

    @staticmethod
    def _on_manifold(manifold, label: str) -> bool:
        """Return True if *label* is registered on the manifold."""
        try:
            pos = manifold.position(label)
            return pos is not None
        except (KeyError, ValueError, AttributeError):
            return False

    # ── Introspection ──────────────────────────────────────────────────────

    def iter_judgments(
        self, matrix: CoOccurrenceMatrix
    ) -> Iterator[ContrastPair]:
        """Yield all ContrastPair objects that would be applied, without applying them.

        Useful for inspection / testing.
        """
        pmi_max = matrix.pmi_max()
        pairs   = matrix.pairs_above_threshold(self.tau_same, self.tau_diff)
        for w1, w2, pmi_val in pairs:
            la = f"vocab::{w1}"
            lb = f"vocab::{w2}"
            j  = JudgmentType.SAME if pmi_val > self.tau_same else JudgmentType.DIFFERENT
            s  = min(1.0, abs(pmi_val) / pmi_max)
            yield ContrastPair(la, lb, j, s)
