"""Phase 5 — Full Integration & Evaluation Framework.

Wires all seven components (C1–C7) into a single end-to-end pipeline
and provides a token-free, weight-free evaluation framework.

Sub-packages
------------
pipeline    — GEOPipeline (full C1–C7 end-to-end entry point)
evaluation  — PipelineEvaluator, EvaluationResult, SuiteResult, and metric helpers
"""

from .pipeline.pipeline import GEOPipeline
from .pipeline.result import PipelineResult
from .evaluation.evaluator import PipelineEvaluator
from .evaluation.suite import SuiteResult

__all__ = [
    "GEOPipeline",
    "PipelineResult",
    "PipelineEvaluator",
    "SuiteResult",
]
