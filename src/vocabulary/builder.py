"""Vocabulary Builder — orchestrates the full vocabulary geometry pipeline.

Single entry point that wires together:
  CoOccurrenceCounter  → text stream → PMI matrix
  WordPlacer           → PMI matrix vocabulary → words on M(t)
  ContrastScheduler    → PMI matrix → C4 judgment stream → refined geometry
  TemplateBuilder      → M(t) word positions → ExpressionEntry objects
  VocabularyStore      → ExpressionEntry list → vocabulary.npz

Data flow:
  VocabularyBuilder.feed(text)
    └→ CoOccurrenceCounter accumulates counts

  VocabularyBuilder.build_and_save(path)
    ├→ counter.build()                 → CoOccurrenceMatrix
    ├→ WordPlacer.place_batch()        → vocab::* on M(t)
    ├→ ContrastScheduler.run_passes()  → geometry refined
    ├→ TemplateBuilder.calibrate_phrase_radius()
    ├→ TemplateBuilder.build(matrix)   → List[ExpressionEntry]
    └→ VocabularyStore.save(entries, path)

Design constraints upheld
--------------------------
NO WEIGHTS     — Co-occurrence counts → PMI via log-ratio; no gradient
NO TOKENS      — Text is normalised words; no token IDs or BPE
NO TRAINING    — Continuous C3/C4 pass; no separate offline phase
LOCAL UPDATES  — All manifold writes go through C3/C4 Gaussian kernels
CAUSALITY FIRST— Directed PMI biases causal fiber (dims 64–79)
SEPARATION     — This module (Group A) never calls C5, C6, or C7
"""

from __future__ import annotations

from typing import Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.phase2.living_manifold.manifold import LivingManifold
    from src.phase3.annealing_engine.engine import AnnealingEngine
    from src.phase2.contrast_engine.engine import ContrastEngine

from src.vocabulary.cooccurrence import CoOccurrenceCounter, CoOccurrenceMatrix
from src.vocabulary.word_placer import WordPlacer
from src.vocabulary.contrast_scheduler import ContrastScheduler
from src.vocabulary.template_builder import TemplateBuilder
from src.vocabulary.vocabulary_store import VocabularyStore


class VocabularyBuilder:
    """End-to-end vocabulary geometry pipeline orchestrator.

    Parameters
    ----------
    manifold         : LivingManifold — the live M(t)
    annealing_engine : AnnealingEngine — C3 engine
    contrast_engine  : ContrastEngine  — C4 engine
    window_size      : co-occurrence window (default 5)
    min_count        : minimum word frequency for inclusion (default 5)
    v_max            : vocabulary ceiling (default 100_000)
    tau_same         : PMI threshold for SAME judgment (default +1.0)
    tau_diff         : PMI threshold for DIFFERENT judgment (default -0.5)
    batch_size       : judgment batch size for ContrastScheduler (default 256)
    n_contrast_passes: number of contrast refinement passes (default 3)
    delta_causal     : minimum dPMI asymmetry for causal bias (default 0.5)
    """

    def __init__(
        self,
        manifold:          "LivingManifold",
        annealing_engine:  "AnnealingEngine",
        contrast_engine:   "ContrastEngine",
        window_size:        int   = 5,
        min_count:          int   = 5,
        v_max:              int   = 100_000,
        tau_same:           float = 1.0,
        tau_diff:           float = -0.5,
        batch_size:         int   = 256,
        n_contrast_passes:  int   = 3,
        delta_causal:       float = 0.5,
    ) -> None:
        self._manifold         = manifold
        self._annealing        = annealing_engine
        self._contrast         = contrast_engine
        self.n_contrast_passes = n_contrast_passes

        # Sub-components
        self._counter = CoOccurrenceCounter(
            window_size=window_size,
            min_count=min_count,
            v_max=v_max,
        )
        self._placer = WordPlacer(annealing_engine)
        self._scheduler = ContrastScheduler(
            contrast_engine,
            tau_same=tau_same,
            tau_diff=tau_diff,
            batch_size=batch_size,
            delta_causal=delta_causal,
        )

        self._matrix: Optional[CoOccurrenceMatrix] = None
        self._n_words_placed: int = 0
        self._n_judgments:    int = 0

    # ── Ingestion ──────────────────────────────────────────────────────────

    def feed(self, text_or_stream) -> "VocabularyBuilder":
        """Ingest text.

        Accepts:
          - A single string (one article, sentence, paragraph)
          - An iterable of strings (multiple texts)

        Returns self for chaining.
        """
        if isinstance(text_or_stream, str):
            self._counter.feed(text_or_stream)
        else:
            self._counter.feed_stream(text_or_stream)
        return self

    # ── Build ──────────────────────────────────────────────────────────────

    def build_and_save(self, path: str) -> int:
        """Run the full pipeline and write vocabulary to *path*.

        Steps:
          1. Build PMI matrix from accumulated text
          2. Place all vocabulary words on M(t) via C3
          3. Run contrast passes via C4 to refine geometry
          4. Calibrate phrase radius
          5. Build ExpressionEntry objects at three levels
          6. Save to .npz

        Parameters
        ----------
        path : output file path (e.g. "vocabulary.npz")

        Returns
        -------
        int — number of entries written to path.
        """
        # ── Step 1: Build PMI matrix ───────────────────────────────────────
        matrix = self._counter.build()
        self._matrix = matrix

        # ── Step 2: Place words on M(t) ────────────────────────────────────
        vocab = matrix.vocabulary
        freq_ranks = list(range(1, len(vocab) + 1))
        self._placer.place_batch(vocab, freq_ranks)
        self._n_words_placed = len(vocab)

        # ── Step 3: Contrast refinement passes ────────────────────────────
        self._n_judgments = self._scheduler.run_passes(
            matrix, n_passes=self.n_contrast_passes
        )

        # ── Step 4: Calibrate phrase radius ───────────────────────────────
        builder = TemplateBuilder(self._manifold)
        builder.calibrate_phrase_radius()

        # ── Step 5: Build entries ──────────────────────────────────────────
        entries = builder.build(matrix)

        if not entries:
            raise RuntimeError(
                "TemplateBuilder produced zero entries.  "
                "Check that words were successfully placed on M(t)."
            )

        # ── Step 6: Save ───────────────────────────────────────────────────
        n_written = VocabularyStore.save(entries, path)
        return n_written

    def build(self) -> List:
        """Build vocabulary entries without saving to disk.

        Returns
        -------
        List[ExpressionEntry]
        """
        matrix = self._counter.build()
        self._matrix = matrix

        vocab      = matrix.vocabulary
        freq_ranks = list(range(1, len(vocab) + 1))
        self._placer.place_batch(vocab, freq_ranks)
        self._n_words_placed = len(vocab)
        self._n_judgments    = self._scheduler.run_passes(
            matrix, n_passes=self.n_contrast_passes
        )

        builder = TemplateBuilder(self._manifold)
        builder.calibrate_phrase_radius()
        return builder.build(matrix)

    # ── Introspection ──────────────────────────────────────────────────────

    @property
    def n_tokens_fed(self) -> int:
        """Total tokens fed to the counter."""
        return self._counter.n_tokens_seen

    @property
    def n_words_placed(self) -> int:
        """Words placed on M(t) after the last build."""
        return self._n_words_placed

    @property
    def n_judgments_applied(self) -> int:
        """Contrast judgments applied in the last build."""
        return self._n_judgments

    @property
    def matrix(self) -> Optional[CoOccurrenceMatrix]:
        """The CoOccurrenceMatrix from the last build, or None."""
        return self._matrix

    def summary(self) -> str:
        """Human-readable build summary."""
        return (
            f"VocabularyBuilder:\n"
            f"  tokens fed        : {self.n_tokens_fed}\n"
            f"  words placed      : {self._n_words_placed}\n"
            f"  contrast judgments: {self._n_judgments}\n"
            f"  matrix size       : "
            f"{len(self._matrix.vocabulary) if self._matrix else 0} words"
        )
