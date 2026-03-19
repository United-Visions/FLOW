"""
Resonance Matcher
=================
Matches a wave segment to the linguistic expression whose semantic wave
most faithfully traces it.

Architecture role
-----------------
This is Stage 2 of the Expression Renderer pipeline:

    Stage 1: Segmentation   →  WaveSegment list
    Stage 2: ResonanceMatcher  ←  THIS FILE
    Stage 3: Flow Preservation →  final language

The core operation (from the spec):
--------------------------------------
For each segment Ψᵢ, find the expression E that minimises:

    resonance_distance(Ψᵢ, semantic_wave(E))

Where:
    semantic_wave(E) = the resonance profile of expression E
                       approximated by running E through a lightweight
                       forward model of the manifold

This is constraint satisfaction, not token prediction:
    - Many candidate expressions evaluated simultaneously
    - The one whose meaning-wave best matches Ψᵢ is selected
    - Length, register, and complexity emerge from Ψᵢ's structure

Prototype approach
------------------
In Phase 1b we use a curated vocabulary of expressions, each annotated
with a semantic wave profile (a vector in the same space as Ψ).

These semantic wave profiles are derived from:
    1. The expression's structural properties (not learned from a corpus)
    2. Feature extraction from the expression's grammar and semantics
    3. The resonance signature the expression would produce

The vocabulary is deliberately small and diverse.  It is not a lookup table —
expression selection is done via resonance distance minimisation across the
entire vocabulary simultaneously.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .wave import WaveSegment, WAVE_DIM

# Optional: FAISS acceleration for >1K vocabulary entries
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


# ── Expression catalogue entry ────────────────────────────────────────────


@dataclass
class ExpressionEntry:
    """
    A candidate expression with its semantic wave profile.

    Attributes
    ----------
    text            : the expression template (may contain {} slots)
    wave_profile    : WAVE_DIM-dimensional semantic wave fingerprint
    register        : 'formal', 'neutral', 'casual'
    rhythm          : 'short'|'medium'|'long' (maps to flow speed)
    uncertainty_fit : how well this expression handles uncertain content
    causal_strength : how strongly this expression encodes causation
    hedging         : whether this expression introduces hedging/tentativeness
    """
    text:             str
    wave_profile:     np.ndarray
    register:         str   = "neutral"
    rhythm:           str   = "medium"
    uncertainty_fit:  float = 0.5
    causal_strength:  float = 0.0
    hedging:          bool  = False


@dataclass
class MatchResult:
    """
    The result of matching a wave segment to an expression.

    Attributes
    ----------
    expression      : the best-matching ExpressionEntry
    resonance_score : similarity score (1 = perfect match, 0 = no match)
    alternatives    : next-best expressions (for debugging / diversity)
    segment_index   : which segment this result belongs to
    """
    expression:      ExpressionEntry
    resonance_score: float
    alternatives:    List[ExpressionEntry] = field(default_factory=list)
    segment_index:   int = 0

    def __repr__(self) -> str:
        return (
            f"MatchResult(i={self.segment_index}, "
            f"score={self.resonance_score:.3f}, "
            f"text='{self.expression.text[:60]}')"
        )


class ResonanceMatcher:
    """
    Matches wave segments to linguistic expressions via resonance distance.

    The matcher maintains a vocabulary of candidate expressions, each with
    a pre-computed semantic wave profile.  Matching is pure constraint
    satisfaction: find the expression whose profile minimises the distance
    to the segment's wave.

    No token prediction.  No left-to-right generation.  No autoregression.
    All candidates are evaluated simultaneously.
    """

    def __init__(self, dim: int = WAVE_DIM) -> None:
        self.dim = dim
        self.vocabulary: List[ExpressionEntry] = []
        self._faiss_index = None  # built lazily when vocab > threshold
        self._faiss_dirty: bool = True
        self._faiss_threshold: int = 200  # use FAISS when vocab > this
        self._build_vocabulary()

    # ------------------------------------------------------------------ #
    # FAISS index management                                               #
    # ------------------------------------------------------------------ #

    def _ensure_faiss_index(self) -> None:
        """Rebuild the FAISS inner-product index if dirty and FAISS available."""
        if not _HAS_FAISS:
            return
        if not self._faiss_dirty:
            return
        if len(self.vocabulary) < self._faiss_threshold:
            self._faiss_index = None
            self._faiss_dirty = False
            return

        # Build a normalised matrix for cosine similarity via inner product
        n = len(self.vocabulary)
        matrix = np.zeros((n, self.dim), dtype=np.float32)
        for i, entry in enumerate(self.vocabulary):
            wp = entry.wave_profile[:self.dim].astype(np.float32)
            norm = np.linalg.norm(wp)
            if norm > 1e-12:
                wp = wp / norm
            matrix[i] = wp

        index = faiss.IndexFlatIP(self.dim)
        index.add(matrix)
        self._faiss_index = index
        self._faiss_dirty = False

    # ------------------------------------------------------------------ #
    # Vocabulary loading                                                   #
    # ------------------------------------------------------------------ #

    def load_vocabulary(self, path: str) -> int:
        """Load geometrically-derived entries from a VocabularyStore .npz file.

        Appends loaded entries to ``self.vocabulary`` without replacing the
        32 hand-crafted entries already present (those serve as a bootstrapping
        floor and are kept permanently).

        Parameters
        ----------
        path : str
            Path to a .npz file created by ``VocabularyStore.save()``.

        Returns
        -------
        int — number of entries loaded and appended.

        Raises
        ------
        FileNotFoundError if *path* does not exist.
        """
        # Lazy import to avoid circular dependency at module load time.
        from src.vocabulary.vocabulary_store import VocabularyStore
        entries = VocabularyStore.load(path)
        self.vocabulary.extend(entries)
        self._faiss_dirty = True
        return len(entries)

    # ------------------------------------------------------------------ #
    # Core matching                                                          #
    # ------------------------------------------------------------------ #

    def match(
        self,
        segment: WaveSegment,
        n_alternatives: int = 3,
        recently_used: Optional[List[str]] = None,
    ) -> MatchResult:
        """
        Find the expression whose semantic wave best matches the segment.

        Process
        -------
        1. Compute the segment's aggregate wave vector (weighted mean of points)
        2. Compute resonance distance to every vocabulary entry
        3. Select the minimum — subject to coherence and uncertainty constraints
        4. Return the match with alternatives

        Parameters
        ----------
        segment        : a WaveSegment to match
        n_alternatives : how many runner-up expressions to return
        recently_used  : list of recently-selected template texts; entries
                         appearing here receive a diversity penalty so the
                         renderer varies its sentence structures.

        Returns
        -------
        MatchResult
        """
        if not self.vocabulary:
            raise RuntimeError("Vocabulary is empty — cannot match.")

        if recently_used is None:
            recently_used = []

        # Step 1: Aggregate the segment into a single representative vector
        query_vec = self._aggregate_segment(segment)

        # Step 2: Score vocabulary entries — use FAISS pre-selection if available
        scores: List[Tuple[float, ExpressionEntry]] = []
        candidates = self._get_candidates(query_vec, n_candidates=50)

        for entry in candidates:
            dist  = self._resonance_distance(query_vec, entry.wave_profile)
            # Penalty for poor uncertainty fit
            uncertainty_penalty = abs(entry.uncertainty_fit - segment.uncertainty) * 0.3
            # Penalty for rhythm mismatch vs flow speed
            rhythm_penalty = self._rhythm_penalty(entry.rhythm, segment.flow_speed) * 0.2
            # Diversity penalty: penalise recently-used templates so consecutive
            # segments don't repeat the same structure.
            recency_count = recently_used.count(entry.text)
            diversity_penalty = 0.15 * recency_count
            total_dist = dist + uncertainty_penalty + rhythm_penalty + diversity_penalty
            scores.append((total_dist, entry))

        # Step 3: Sort by ascending distance (lower = better match)
        scores.sort(key=lambda x: x[0])

        best_dist, best_expr = scores[0]
        alts = [e for _, e in scores[1: 1 + n_alternatives]]

        # Convert distance to similarity score: sim = exp(-dist)
        resonance_score = float(np.exp(-best_dist))

        return MatchResult(
            expression=best_expr,
            resonance_score=resonance_score,
            alternatives=alts,
            segment_index=segment.index,
        )

    def match_all(
        self, segments: List[WaveSegment], n_alternatives: int = 2
    ) -> List[MatchResult]:
        """Match all segments, applying a diversity penalty so consecutive
        segments don't reuse the same expression template."""
        results: List[MatchResult] = []
        recently_used: List[str] = []   # sliding window of last 3 templates
        for seg in segments:
            result = self.match(seg, n_alternatives, recently_used=recently_used[-3:])
            results.append(result)
            recently_used.append(result.expression.text)
        return results

    # ------------------------------------------------------------------ #
    # Candidate pre-selection                                              #
    # ------------------------------------------------------------------ #

    def _get_candidates(
        self, query_vec: np.ndarray, n_candidates: int = 50
    ) -> List[ExpressionEntry]:
        """Return candidate entries for scoring.

        If FAISS is available and the vocabulary is large, use approximate
        nearest-neighbour search to pre-select the top-n candidates.
        Otherwise fall back to the full vocabulary (linear scan).
        """
        self._ensure_faiss_index()

        if self._faiss_index is not None and len(self.vocabulary) > self._faiss_threshold:
            # FAISS inner-product search on normalised vectors → cosine similarity
            qv = query_vec[:self.dim].astype(np.float32)
            qnorm = np.linalg.norm(qv)
            if qnorm > 1e-12:
                qv = qv / qnorm
            k = min(n_candidates, len(self.vocabulary))
            _, idxs = self._faiss_index.search(qv.reshape(1, -1), k)
            return [self.vocabulary[int(i)] for i in idxs[0] if 0 <= i < len(self.vocabulary)]

        # Fallback: return full vocabulary for linear scan
        return list(self.vocabulary)

    # ------------------------------------------------------------------ #
    # Distance metric                                                        #
    # ------------------------------------------------------------------ #

    def _resonance_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Resonance distance between two semantic wave profiles.

        Combines:
        - Cosine distance (direction of meaning)
        - L2 norm difference (intensity of meaning)

        Both are informative: same direction with different magnitude
        = same theme, different emphasis — that is a near-match, not a
        perfect match.
        """
        n1 = np.linalg.norm(v1) + 1e-12
        n2 = np.linalg.norm(v2) + 1e-12
        cosine_sim  = float(np.dot(v1, v2) / (n1 * n2))
        cosine_dist = 1.0 - cosine_sim

        intensity_gap = abs(n1 - n2) / (n1 + n2)

        return float(0.7 * cosine_dist + 0.3 * intensity_gap)

    def _aggregate_segment(self, segment: WaveSegment) -> np.ndarray:
        """
        Compute the representative vector for a wave segment.

        Amplitude-weighted centroid of the segment's wave points.
        Points with higher amplitude contribute more to the representation.
        """
        if not segment.points:
            return np.zeros(self.dim)

        vecs  = np.stack([p.vector[:self.dim] for p in segment.points])
        amps  = np.array([p.amplitude for p in segment.points])
        amps /= (amps.sum() + 1e-12)
        return vecs.T @ amps   # weighted sum

    def _rhythm_penalty(self, rhythm: str, flow_speed: float) -> float:
        """
        Penalty for rhythm mismatch relative to flow speed.

        fast flow (0.8+) → prefers short expressions
        slow flow (0.3-) → prefers long expressions
        """
        target = flow_speed   # 0=long 1=short (inverted)
        rhythm_map = {"short": 0.8, "medium": 0.5, "long": 0.2}
        rhythm_val = rhythm_map.get(rhythm, 0.5)
        return abs(target - rhythm_val)

    # ------------------------------------------------------------------ #
    # Vocabulary construction                                               #
    # ------------------------------------------------------------------ #

    def _build_vocabulary(self) -> None:
        """
        Build the expression vocabulary with semantic wave profiles.

        Each entry's wave_profile is a structured vector in WAVE_DIM space,
        derived from the expression's semantic and grammatical features.
        Not learned from a corpus — derived from structural properties.
        """
        entries_raw = self._expression_templates()
        for (text, features) in entries_raw:
            profile = self._compute_wave_profile(features)
            entry = ExpressionEntry(
                text=text,
                wave_profile=profile,
                register=features.get("register", "neutral"),
                rhythm=features.get("rhythm", "medium"),
                uncertainty_fit=features.get("uncertainty_fit", 0.5),
                causal_strength=features.get("causal_strength", 0.0),
                hedging=features.get("hedging", False),
            )
            self.vocabulary.append(entry)

    def _compute_wave_profile(self, features: dict) -> np.ndarray:
        """
        Compute a semantic wave profile from structural features.

        The profile is a structured vector capturing:
        - Axes 0-15    : causal signal strength
        - Axes 16-31   : logical structure signal
        - Axes 32-47   : uncertainty/hedging signal
        - Axes 48-63   : temporal/sequential signal
        - Axes 64-79   : contrast/comparison signal
        - Axes 80-95   : emphasis/confirmation signal
        - Axes 96-103  : meta-communicative signal
        """
        profile = np.zeros(self.dim)

        # Causal signal (axes 0-15)
        causal = float(features.get("causal_strength", 0.0))
        rng = np.random.default_rng(hash(str(features)) % 2**31)
        profile[0:16]  += causal  * (0.5 + 0.5 * rng.uniform(size=16))

        # Uncertainty signal (axes 32-47)
        hedging = float(features.get("hedging", False))
        unc_fit = float(features.get("uncertainty_fit", 0.5))
        profile[32:48] += unc_fit * (0.5 + 0.5 * rng.uniform(size=16))
        if hedging:
            profile[32:48] *= 1.5

        # Rhythm / temporal signal (axes 48-63)
        rhythm_val = {"short": 0.8, "medium": 0.5, "long": 0.2}.get(
            features.get("rhythm", "medium"), 0.5
        )
        profile[48:64] += rhythm_val * rng.uniform(size=16)

        # Logical / structural signal (axes 16-31)
        logical = float(features.get("logical_strength", 0.3))
        profile[16:32] += logical * rng.uniform(size=16)

        # Contrast signal (axes 64-79)
        contrast = float(features.get("contrast", 0.0))
        profile[64:80] += contrast * rng.uniform(size=16)

        # Emphasis signal (axes 80-95)
        emphasis = float(features.get("emphasis", 0.3))
        profile[80:96] += emphasis * rng.uniform(size=16)

        # Meta-communicative (axes 96-103)
        meta = float(features.get("meta", 0.0))
        profile[96:self.dim] += meta * rng.uniform(size=self.dim - 96)

        # Normalise
        norm = np.linalg.norm(profile)
        if norm > 1e-12:
            profile /= norm

        return profile

    def _expression_templates(self) -> list:
        """
        The curated expression vocabulary.

        Each entry is (text_template, feature_dict).
        Features determine the expression's semantic wave profile.
        """
        return [
            # ── Causal / mechanistic ──────────────────────────────────
            ("This happens because {}.",
             {"causal_strength": 0.9, "logical_strength": 0.5, "rhythm": "medium",
              "uncertainty_fit": 0.2, "register": "neutral"}),

            ("The mechanism is: {}.",
             {"causal_strength": 0.95, "logical_strength": 0.7, "rhythm": "medium",
              "uncertainty_fit": 0.1, "register": "formal"}),

            ("{} causes {} through {}.",
             {"causal_strength": 1.0, "logical_strength": 0.6, "rhythm": "medium",
              "uncertainty_fit": 0.1, "register": "formal"}),

            ("As a result, {}.",
             {"causal_strength": 0.8, "logical_strength": 0.4, "rhythm": "short",
              "uncertainty_fit": 0.2, "register": "neutral"}),

            ("This leads directly to {}.",
             {"causal_strength": 0.85, "logical_strength": 0.4, "rhythm": "short",
              "uncertainty_fit": 0.2, "register": "neutral"}),

            ("The chain of events: {}.",
             {"causal_strength": 0.9, "logical_strength": 0.5, "rhythm": "medium",
              "uncertainty_fit": 0.2, "register": "formal"}),

            # ── Uncertainty / hedging ─────────────────────────────────
            ("It appears that {}.",
             {"hedging": True, "uncertainty_fit": 0.8, "causal_strength": 0.1,
              "rhythm": "short", "register": "neutral"}),

            ("The evidence suggests {}.",
             {"hedging": True, "uncertainty_fit": 0.7, "causal_strength": 0.3,
              "rhythm": "medium", "register": "formal"}),

            ("This is likely because {}.",
             {"hedging": True, "uncertainty_fit": 0.75, "causal_strength": 0.5,
              "rhythm": "medium", "register": "neutral"}),

            ("One possibility is that {}.",
             {"hedging": True, "uncertainty_fit": 0.85, "causal_strength": 0.0,
              "rhythm": "medium", "register": "neutral"}),

            ("Under certain conditions, {}.",
             {"hedging": True, "uncertainty_fit": 0.8, "causal_strength": 0.3,
              "rhythm": "medium", "register": "formal"}),

            # ── Contrast / comparison ─────────────────────────────────
            ("However, {}.",
             {"contrast": 0.9, "causal_strength": 0.1, "rhythm": "short",
              "uncertainty_fit": 0.3, "register": "neutral"}),

            ("Unlike {}, the {} is different in that {}.",
             {"contrast": 1.0, "causal_strength": 0.1, "rhythm": "long",
              "uncertainty_fit": 0.3, "register": "neutral"}),

            ("The key distinction is {}.",
             {"contrast": 0.85, "logical_strength": 0.7, "rhythm": "medium",
              "uncertainty_fit": 0.2, "register": "formal"}),

            ("While {} is true, {} follows from different reasoning.",
             {"contrast": 0.9, "logical_strength": 0.6, "rhythm": "long",
              "uncertainty_fit": 0.3, "register": "formal"}),

            # ── Logical / structural ─────────────────────────────────
            ("Therefore, {}.",
             {"logical_strength": 0.9, "causal_strength": 0.4, "rhythm": "short",
              "uncertainty_fit": 0.1, "register": "formal"}),

            ("If {} then {}.",
             {"logical_strength": 1.0, "causal_strength": 0.5, "rhythm": "medium",
              "uncertainty_fit": 0.2, "register": "neutral"}),

            ("Given that {}, it follows that {}.",
             {"logical_strength": 0.95, "causal_strength": 0.5, "rhythm": "medium",
              "uncertainty_fit": 0.1, "register": "formal"}),

            ("Both {} and {} share {}.",
             {"logical_strength": 0.7, "contrast": 0.5, "rhythm": "medium",
              "uncertainty_fit": 0.3, "register": "neutral"}),

            # ── Descriptive / observational ───────────────────────────
            ("{}.",
             {"logical_strength": 0.3, "causal_strength": 0.1, "rhythm": "short",
              "uncertainty_fit": 0.3, "register": "neutral"}),

            ("The {} is characterised by {}.",
             {"logical_strength": 0.5, "causal_strength": 0.1, "rhythm": "medium",
              "uncertainty_fit": 0.3, "register": "formal"}),

            ("Consider {}.",
             {"logical_strength": 0.3, "meta": 0.5, "rhythm": "short",
              "uncertainty_fit": 0.5, "register": "neutral"}),

            ("This involves {}.",
             {"logical_strength": 0.4, "causal_strength": 0.3, "rhythm": "short",
              "uncertainty_fit": 0.4, "register": "neutral"}),

            # ── Emphasis / conclusion ─────────────────────────────────
            ("Critically, {}.",
             {"emphasis": 0.9, "logical_strength": 0.5, "rhythm": "short",
              "uncertainty_fit": 0.1, "register": "formal"}),

            ("In summary: {}.",
             {"emphasis": 0.8, "logical_strength": 0.6, "rhythm": "medium",
              "uncertainty_fit": 0.1, "register": "neutral", "meta": 0.5}),

            ("The key insight is that {}.",
             {"emphasis": 0.9, "logical_strength": 0.6, "rhythm": "medium",
              "uncertainty_fit": 0.15, "register": "formal"}),

            ("This demonstrates that {}.",
             {"emphasis": 0.85, "causal_strength": 0.4, "rhythm": "medium",
              "uncertainty_fit": 0.1, "register": "formal"}),

            # ── Sequential / procedural ───────────────────────────────
            ("First, {}. Then, {}.",
             {"logical_strength": 0.7, "causal_strength": 0.5, "rhythm": "medium",
              "uncertainty_fit": 0.2, "register": "neutral"}),

            ("The process unfolds as follows: {}.",
             {"logical_strength": 0.6, "causal_strength": 0.6, "rhythm": "long",
              "uncertainty_fit": 0.2, "register": "formal"}),

            ("Beginning with {}, this leads through {} to {}.",
             {"logical_strength": 0.6, "causal_strength": 0.8, "rhythm": "long",
              "uncertainty_fit": 0.2, "register": "neutral"}),

            # ── Warning / advisory ────────────────────────────────────
            ("Importantly, {}.",
             {"emphasis": 0.8, "meta": 0.3, "rhythm": "short",
              "uncertainty_fit": 0.2, "register": "formal"}),

            ("One must be careful that {}.",
             {"emphasis": 0.7, "uncertainty_fit": 0.4, "rhythm": "medium",
              "register": "formal"}),
        ]
