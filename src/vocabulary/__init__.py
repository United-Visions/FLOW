"""src/vocabulary — Vocabulary Geometry module (Phase 7).

This module implements Group A (manifold shaping) vocabulary growth for the
Geometric Causal Architecture.  It extends the C7 expression vocabulary from
32 hand-crafted templates to ~100,000 geometrically-grounded entries.

Public API
----------
VocabularyBuilder     — end-to-end orchestrator (main entry point)
CoOccurrenceCounter   — text stream → PMI matrix
WordPlacer            — word string → manifold position via C3
ContrastScheduler     — PMI matrix → C4 judgment stream
TemplateBuilder       — manifold positions → ExpressionEntry objects
VocabularyStore       — ExpressionEntry ↔ .npz serialisation

No C5, C6, or C7 components are imported here.  This module is purely
Group A (manifold shaping).  The only external contract it satisfies is:
  - Words appear on M(t) under label ``"vocab::{word}"``
  - ExpressionEntry objects are saved to / loaded from vocabulary.npz
  - ResonanceMatcher.load_vocabulary() reads those entries
"""

from .cooccurrence import CoOccurrenceCounter, CoOccurrenceMatrix
from .word_placer import WordPlacer, structural_feature_vector
from .contrast_scheduler import ContrastScheduler, ContrastPair, CausalBiasDirective
from .template_builder import TemplateBuilder, compose_wave_profile
from .vocabulary_store import VocabularyStore
from .builder import VocabularyBuilder

__all__ = [
    "VocabularyBuilder",
    "CoOccurrenceCounter",
    "CoOccurrenceMatrix",
    "WordPlacer",
    "structural_feature_vector",
    "ContrastScheduler",
    "ContrastPair",
    "CausalBiasDirective",
    "TemplateBuilder",
    "compose_wave_profile",
    "VocabularyStore",
]
