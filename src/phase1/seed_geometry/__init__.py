"""
Seed Geometry Engine — Component 1

Derives the initial mathematical skeleton of the manifold from first principles.
Runs once. Output (M₀) is static forever.

Sub-modules
-----------
causal          — directed Riemannian structure from Pearl's do-calculus
logical         — Boolean hypercube topology from propositional logic
probabilistic   — statistical simplex with Fisher metric from Kolmogorov axioms
similarity      — metric space with flexible local curvature
composer        — fiber bundle composition of all four geometries
engine          — public SeedGeometryEngine + SeedManifold types
"""

from .engine import SeedGeometryEngine
from .manifold import SeedManifold, ManifoldPoint

__all__ = ["SeedGeometryEngine", "SeedManifold", "ManifoldPoint"]
