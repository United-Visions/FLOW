"""Evaluation sub-package — geometry-grounded pipeline evaluation."""

from .metrics import CoherenceMetrics, CausalMetrics, LocalityMetrics, EvaluationResult
from .evaluator import PipelineEvaluator
from .suite import SuiteResult

__all__ = [
    "CoherenceMetrics",
    "CausalMetrics",
    "LocalityMetrics",
    "EvaluationResult",
    "PipelineEvaluator",
    "SuiteResult",
]
