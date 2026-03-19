"""Phase 2 — The Living Manifold.

Components:
  2a. LivingManifold  — dynamic Riemannian manifold data structure
  2b. ContrastEngine  — same/different relational placement operator
"""

from .living_manifold import LivingManifold, RegionType
from .contrast_engine import ContrastEngine, JudgmentType

__all__ = ["LivingManifold", "RegionType", "ContrastEngine", "JudgmentType"]
