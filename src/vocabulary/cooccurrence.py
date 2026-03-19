"""Co-occurrence Counter — converts a text stream into a PMI matrix.

This is the first component of the vocabulary geometry pipeline.

It is NOT a tokeniser.  Words are treated as geometric events — raw
linguistic observations that produce evidence about semantic proximity.
The co-occurrence window is a temporal locality window on a text stream,
not a symbol-ID array.

The only output that survives into later phases is the (w₁, w₂, PMI) triple.
Raw counts are held in memory during build() and then discarded.

Mathematics
-----------
PMI(w₁, w₂)   = log[ P(w₁,w₂) / (P(w₁)·P(w₂)) ]
dPMI(w₁→w₂)   = log[ P(w₂|w₁)  / P(w₂) ]

Both are computed from raw sliding-window counts — no smoothing, no
dimensionality reduction, no model parameters.
"""

from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# CoOccurrenceMatrix — result of a completed counting session
# ─────────────────────────────────────────────────────────────────────────────

class CoOccurrenceMatrix:
    """Holds PMI and directed PMI values for a completed vocabulary.

    Attributes
    ----------
    vocabulary      : ordered list of accepted words (above min_count)
    _pmi            : dict (w1,w2) → symmetric PMI float
    _dpmi           : dict (w1,w2) → directed PMI float  (w1 precedes w2)
    _unigram_counts : dict word → total unigram count
    """

    def __init__(
        self,
        pmi_dict: Dict[Tuple[str, str], float],
        dpmi_dict: Dict[Tuple[str, str], float],
        unigram_counts: Dict[str, int],
        vocabulary: List[str],
    ) -> None:
        self._pmi = pmi_dict
        self._dpmi = dpmi_dict
        self._unigram_counts = unigram_counts
        self.vocabulary: List[str] = vocabulary

    # ── Accessors ──────────────────────────────────────────────────────────

    def pmi(self, w1: str, w2: str) -> float:
        """Symmetric PMI between two words.  Returns 0.0 if pair not seen."""
        key = (min(w1, w2), max(w1, w2))
        return float(self._pmi.get(key, 0.0))

    def dpmi(self, w1: str, w2: str) -> float:
        """Directed PMI: how much more likely w2 is given w1 precedes it.
        Returns 0.0 if the directed pair was not seen."""
        return float(self._dpmi.get((w1, w2), 0.0))

    def frequency_rank(self, word: str) -> int:
        """1-based frequency rank of a word (rank 1 = most frequent).
        Returns len(vocabulary)+1 for unknown words."""
        sorted_vocab = sorted(
            self._unigram_counts.items(), key=lambda x: -x[1]
        )
        rank_map = {w: i + 1 for i, (w, _) in enumerate(sorted_vocab)}
        return rank_map.get(word, len(self.vocabulary) + 1)

    def unigram_count(self, word: str) -> int:
        """Raw unigram count of a word."""
        return int(self._unigram_counts.get(word, 0))

    def pairs_above_threshold(
        self, tau_same: float, tau_diff: float
    ) -> List[Tuple[str, str, float]]:
        """Return all (w1, w2, pmi) triples where |pmi| exceeds a threshold.

        Only pairs where pmi > tau_same or pmi < tau_diff are returned.
        Pairs in the neutral zone are excluded.

        Returns
        -------
        list of (w1, w2, pmi) — sorted by abs(pmi) descending.
        """
        result = []
        for (w1, w2), val in self._pmi.items():
            if val > tau_same or val < tau_diff:
                result.append((w1, w2, val))
        result.sort(key=lambda x: -abs(x[2]))
        return result

    def directed_pairs_above_delta(
        self, delta: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """Return directed pairs where dPMI(w1→w2) exceeds dPMI(w2→w1) by delta.

        Returns
        -------
        list of (w1, w2, dpmi_diff) sorted by dpmi_diff descending.
        """
        result = []
        seen: Set[Tuple[str, str]] = set()
        for (w1, w2), val in self._dpmi.items():
            if (w2, w1) in seen:
                continue
            seen.add((w1, w2))
            reverse = self._dpmi.get((w2, w1), 0.0)
            diff = val - reverse
            if diff > delta:
                result.append((w1, w2, diff))
        result.sort(key=lambda x: -x[2])
        return result

    def pmi_max(self) -> float:
        """Maximum absolute PMI value across all pairs."""
        if not self._pmi:
            return 1.0
        return max(abs(v) for v in self._pmi.values()) or 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CoOccurrenceCounter — incremental stream processor
# ─────────────────────────────────────────────────────────────────────────────

class CoOccurrenceCounter:
    """Convert a raw text stream into a PMI matrix.

    This is NOT a tokeniser.  Words are normalised (lowercase, stripped
    punctuation) and co-occurrence counts are accumulated over a sliding
    context window.  No word IDs, no special tokens, no padding.

    Parameters
    ----------
    window_size : int
        How many words on each side of a target word form its context.
        Default 5 (standard linguistic co-occurrence window).
    min_count : int
        Words seen fewer than this many times are pruned before PMI
        computation.  Geometric placement of hapax legomena is unreliable.
        Default 5.
    v_max : int
        Hard ceiling on vocabulary size.  The top-v_max most frequent
        words (above min_count) are retained, others pruned.
        Default 100_000.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_count:   int = 5,
        v_max:       int = 100_000,
    ) -> None:
        self.window_size = window_size
        self.min_count   = min_count
        self.v_max       = v_max

        # Raw counts — discarded after build()
        self._unigram:   Counter[str]                     = Counter()
        self._bigram:    Counter[Tuple[str, str]]         = Counter()
        self._directed:  Counter[Tuple[str, str]]         = Counter()
        self._n_tokens:  int                              = 0
        self._n_pairs:   int                              = 0
        self._built:     bool                             = False

    # ── Ingestion ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> List[str]:
        """Lowercase and strip all non-alphabetic characters, split on whitespace.

        This is deliberately minimal — no stemming, no lemmatisation, no
        decomposition.  Words are geometric events, not symbol stubs.
        """
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return [w for w in text.split() if w]

    def feed(self, text: str) -> None:
        """Ingest one text string (sentence, paragraph, article body).

        Can be called repeatedly.  Accumulates counts incrementally.
        Thread-safety is not guaranteed — call sequentially.

        Parameters
        ----------
        text : any plain-text string; HTML is NOT stripped here.
        """
        if self._built:
            raise RuntimeError(
                "Counter has already been built.  Create a new instance to feed more text."
            )
        words = self._normalise(text)
        n = len(words)
        self._n_tokens += n

        for i, w in enumerate(words):
            self._unigram[w] += 1
            end = min(i + self.window_size + 1, n)
            for j in range(i + 1, end):
                w2 = words[j]
                # Symmetric bigram key — canonical order
                key = (min(w, w2), max(w, w2))
                self._bigram[key] += 1
                self._n_pairs += 1
                # Directed: w precedes w2
                self._directed[(w, w2)] += 1

    def feed_stream(self, texts: Iterable[str]) -> None:
        """Convenience wrapper: calls feed() for each text in the iterable."""
        for text in texts:
            self.feed(text)

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self) -> CoOccurrenceMatrix:
        """Compute PMI and dPMI from accumulated counts.

        Called once after all text has been fed.  Prunes the vocabulary by
        min_count and v_max, computes PMI from the surviving words, then
        discards the raw count arrays to free memory.

        Returns
        -------
        CoOccurrenceMatrix
        """
        if self._built:
            raise RuntimeError("build() may only be called once per instance.")
        self._built = True

        # ── Step 1: Prune vocabulary ──────────────────────────────────────
        # Keep words above min_count, then take top v_max by frequency
        accepted: List[Tuple[str, int]] = [
            (w, c) for w, c in self._unigram.items() if c >= self.min_count
        ]
        accepted.sort(key=lambda x: -x[1])
        accepted = accepted[: self.v_max]
        vocab_set: Set[str] = {w for w, _ in accepted}
        vocabulary: List[str] = [w for w, _ in accepted]

        # Final unigram counts for the accepted vocabulary
        unigram_counts: Dict[str, int] = {
            w: self._unigram[w] for w in vocabulary
        }
        N = sum(unigram_counts.values()) or 1

        # ── Step 2: Compute PMI ───────────────────────────────────────────
        pmi_dict: Dict[Tuple[str, str], float] = {}

        for (w1, w2), co_count in self._bigram.items():
            if w1 not in vocab_set or w2 not in vocab_set:
                continue
            if co_count < 2:          # sub-threshold bigrams pruned early
                continue
            p_w1  = unigram_counts[w1] / N
            p_w2  = unigram_counts[w2] / N
            p_co  = co_count / N
            if p_w1 > 0 and p_w2 > 0 and p_co > 0:
                raw_pmi = math.log(p_co / (p_w1 * p_w2))
                key = (min(w1, w2), max(w1, w2))
                pmi_dict[key] = raw_pmi

        # ── Step 3: Compute directed PMI ──────────────────────────────────
        dpmi_dict: Dict[Tuple[str, str], float] = {}

        for (w1, w2), d_count in self._directed.items():
            if w1 not in vocab_set or w2 not in vocab_set:
                continue
            if d_count < 2:
                continue
            p_w1  = unigram_counts[w1] / N
            p_w2  = unigram_counts[w2] / N
            p_d   = d_count / N
            if p_w1 > 0 and p_w2 > 0 and p_d > 0:
                dpmi_dict[(w1, w2)] = math.log(p_d / (p_w1 * p_w2))

        # ── Discard raw counts ─────────────────────────────────────────────
        self._bigram.clear()
        self._directed.clear()
        # Keep _unigram_counts for frequency_rank() queries

        return CoOccurrenceMatrix(
            pmi_dict=pmi_dict,
            dpmi_dict=dpmi_dict,
            unigram_counts=unigram_counts,
            vocabulary=vocabulary,
        )

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def n_tokens_seen(self) -> int:
        """Total tokens fed so far (before any pruning)."""
        return self._n_tokens

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct words seen so far (before pruning)."""
        return len(self._unigram)
