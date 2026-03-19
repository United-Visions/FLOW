"""Persistence sub-package — ManifoldSnapshot save / load.

Phase 8 — Scaling Priority 2 (unlocks Kaggle builds, HuggingFace Hub,
and all long-running growth jobs).

Serialisation uses .npz (pure numpy arrays), matching the pattern
established by VocabularyStore in Phase 7.  No pickle, no Python
object serialisation.
"""

from .snapshot import ManifoldSnapshot

__all__ = ["ManifoldSnapshot"]
