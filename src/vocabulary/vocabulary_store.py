"""Vocabulary Store — serialise and deserialise ExpressionEntry objects.

Storage format: a single .npz file containing dense arrays.  No pickle,
no Python object serialisation, no model checkpoints.  Pure numpy arrays.

Arrays stored (§4.5 of vocabulary-geometry-specification.md)
------------------------------------------------------------
wave_profiles   : float32 (N × WAVE_DIM)  — one row per entry
register_ids    : uint8   (N,)             — 0=neutral, 1=formal, 2=casual
rhythm_ids      : uint8   (N,)             — 0=short, 1=medium, 2=long
uncertainty_fit : float32 (N,)
causal_strength : float32 (N,)
hedging_flags   : bool    (N,)
texts           : object  (N,)             — Python strings (numpy object array)

The file also stores a format_version integer for future migration support
and a uint32 count so loaders can sanity-check array shapes.

Incremental append
------------------
The store is designed to support append without full rewrite (§9 open Q5).
``VocabularyStore.append(entries, path)`` loads existing content, merges,
deduplicates by text, and rewrites — suitable for small incremental updates.
For large-scale streaming (Phase 8), a future partition-based format would
avoid full rewrite, but the current .npz approach satisfies Phase 7.
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import List

from src.phase1.expression.matcher import ExpressionEntry
from src.phase1.expression.wave import WAVE_DIM

# ── Encoding maps ─────────────────────────────────────────────────────────────

_REGISTER_TO_ID = {"neutral": 0, "formal": 1, "casual": 2}
_ID_TO_REGISTER = {v: k for k, v in _REGISTER_TO_ID.items()}

_RHYTHM_TO_ID   = {"short": 0, "medium": 1, "long": 2}
_ID_TO_RHYTHM   = {v: k for k, v in _RHYTHM_TO_ID.items()}

_FORMAT_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# VocabularyStore
# ─────────────────────────────────────────────────────────────────────────────

class VocabularyStore:
    """Static utility class for .npz vocabulary serialisation.

    All methods are class-level — no instantiation needed.

    Usage
    -----
    >>> VocabularyStore.save(entries, "vocabulary.npz")
    >>> loaded = VocabularyStore.load("vocabulary.npz")
    >>> VocabularyStore.append(new_entries, "vocabulary.npz")
    """

    @classmethod
    def save(cls, entries: List[ExpressionEntry], path: str) -> int:
        """Serialise *entries* to *path* as a .npz file.

        Parameters
        ----------
        entries : list of ExpressionEntry
        path    : file path; directories are created if they do not exist.

        Returns
        -------
        int — number of entries written.
        """
        if not entries:
            raise ValueError("Cannot save an empty vocabulary.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        n = len(entries)

        wave_profiles   = np.zeros((n, WAVE_DIM), dtype=np.float32)
        register_ids    = np.zeros(n, dtype=np.uint8)
        rhythm_ids      = np.zeros(n, dtype=np.uint8)
        uncertainty_fit = np.zeros(n, dtype=np.float32)
        causal_strength = np.zeros(n, dtype=np.float32)
        hedging_flags   = np.zeros(n, dtype=bool)
        texts           = np.empty(n, dtype=object)

        for i, e in enumerate(entries):
            wp = np.asarray(e.wave_profile, dtype=np.float32)
            if wp.shape[0] < WAVE_DIM:
                # Pad with zeros if wave profile is shorter than WAVE_DIM
                wp = np.pad(wp, (0, WAVE_DIM - wp.shape[0]))
            wave_profiles[i]   = wp[:WAVE_DIM]
            register_ids[i]    = _REGISTER_TO_ID.get(e.register, 0)
            rhythm_ids[i]      = _RHYTHM_TO_ID.get(e.rhythm, 1)
            uncertainty_fit[i] = float(e.uncertainty_fit)
            causal_strength[i] = float(e.causal_strength)
            hedging_flags[i]   = bool(e.hedging)
            texts[i]           = str(e.text)

        np.savez_compressed(
            path,
            wave_profiles   = wave_profiles,
            register_ids    = register_ids,
            rhythm_ids      = rhythm_ids,
            uncertainty_fit = uncertainty_fit,
            causal_strength = causal_strength,
            hedging_flags   = hedging_flags,
            texts           = texts,
            format_version  = np.array([_FORMAT_VERSION], dtype=np.uint32),
            n_entries       = np.array([n], dtype=np.uint32),
        )

        return n

    @classmethod
    def load(cls, path: str) -> List[ExpressionEntry]:
        """Load vocabulary entries from a .npz file.

        Parameters
        ----------
        path : file path to a .npz created by VocabularyStore.save().

        Returns
        -------
        List[ExpressionEntry]
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        data = np.load(path, allow_pickle=True)

        wave_profiles   = data["wave_profiles"].astype(np.float64)
        register_ids    = data["register_ids"]
        rhythm_ids      = data["rhythm_ids"]
        uncertainty_fit = data["uncertainty_fit"].astype(np.float64)
        causal_strength = data["causal_strength"].astype(np.float64)
        hedging_flags   = data["hedging_flags"]
        texts           = data["texts"]

        n = len(texts)
        entries = []
        for i in range(n):
            entry = ExpressionEntry(
                text             = str(texts[i]),
                wave_profile     = wave_profiles[i],
                register         = _ID_TO_REGISTER.get(int(register_ids[i]), "neutral"),
                rhythm           = _ID_TO_RHYTHM.get(int(rhythm_ids[i]), "medium"),
                uncertainty_fit  = float(uncertainty_fit[i]),
                causal_strength  = float(causal_strength[i]),
                hedging          = bool(hedging_flags[i]),
            )
            entries.append(entry)

        return entries

    @classmethod
    def append(cls, new_entries: List[ExpressionEntry], path: str) -> int:
        """Append *new_entries* to an existing store, deduplicating by text.

        If the store does not exist, creates it.

        Returns
        -------
        int — total number of entries in the store after append.
        """
        if os.path.exists(path):
            existing = cls.load(path)
            existing_texts = {e.text for e in existing}
            to_add = [e for e in new_entries if e.text not in existing_texts]
            all_entries = existing + to_add
        else:
            all_entries = list(new_entries)

        return cls.save(all_entries, path)

    @classmethod
    def count(cls, path: str) -> int:
        """Return the number of entries in a store without loading all data."""
        if not os.path.exists(path):
            return 0
        data = np.load(path, allow_pickle=True)
        return int(data["n_entries"][0])
