"""
Expression Renderer — Component 7 (Phase 1b prototype)

Converts the standing wave Ψ into fluent natural language.
This component has NO access to the manifold.
It receives only Ψ and produces language.

Sub-modules
-----------
wave        — StandingWave data type + mock wave constructors
matcher     — ResonanceMatcher (constraint-satisfaction expression selection)
renderer    — ExpressionRenderer (full rendering pipeline: segment → match → flow)
"""

from .wave import StandingWave, WaveSegment, create_mock_wave
from .matcher import ResonanceMatcher
from .renderer import ExpressionRenderer

__all__ = [
    "StandingWave",
    "WaveSegment",
    "create_mock_wave",
    "ResonanceMatcher",
    "ExpressionRenderer",
]
