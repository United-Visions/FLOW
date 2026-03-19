"""SuiteResult — aggregate output from a PipelineEvaluator.run_suite() call.

Holds the per-query EvaluationResults plus summary statistics computed
over the entire suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .metrics import EvaluationResult, CausalMetrics, LocalityMetrics


@dataclass
class SuiteResult:
    """Aggregate result from running the full evaluation suite.

    Attributes
    ----------
    results             : per-query EvaluationResult list
    causal              : CausalMetrics if causal evaluation was run, else None
    locality            : LocalityMetrics if locality check was run, else None
    novelty_decay       : list of novelty scores over repeated exposure (should ↓)
    extra               : dict of additional scalar summary metrics
    """

    results: List[EvaluationResult] = field(default_factory=list)
    causal: Optional[CausalMetrics] = None
    locality: Optional[LocalityMetrics] = None
    novelty_decay: List[float] = field(default_factory=list)
    extra: Dict[str, float] = field(default_factory=dict)

    # ── Summary statistics ─────────────────────────────────────────────────

    @property
    def n_queries(self) -> int:
        """Number of individual queries evaluated."""
        return len(self.results)

    @property
    def mean_coherence(self) -> float:
        """Mean overall coherence score across all queries."""
        if not self.results:
            return 0.0
        return float(np.mean([r.overall_score() for r in self.results]))

    @property
    def mean_render_confidence(self) -> float:
        """Mean C7 rendering confidence across all queries."""
        if not self.results:
            return 0.0
        return float(np.mean([r.coherence.render_confidence for r in self.results]))

    @property
    def mean_wave_confidence(self) -> float:
        """Mean C6 wave confidence across all queries."""
        if not self.results:
            return 0.0
        return float(np.mean([r.coherence.wave_confidence for r in self.results]))

    @property
    def mean_steps(self) -> float:
        """Mean number of SDE steps per query."""
        if not self.results:
            return 0.0
        return float(np.mean([r.coherence.trajectory_steps for r in self.results]))

    @property
    def termination_distribution(self) -> Dict[str, int]:
        """Count of each termination reason across all queries."""
        dist: Dict[str, int] = {}
        for r in self.results:
            reason = r.coherence.termination_reason
            dist[reason] = dist.get(reason, 0) + 1
        return dist

    @property
    def novelty_is_decaying(self) -> bool:
        """True if novelty scores are monotonically decreasing over repetitions."""
        if len(self.novelty_decay) < 2:
            return False
        return all(
            self.novelty_decay[i] >= self.novelty_decay[i + 1]
            for i in range(len(self.novelty_decay) - 1)
        )

    def as_dict(self) -> Dict[str, object]:
        """Return summary statistics as a flat dict for logging / display."""
        d: Dict[str, object] = {
            "n_queries": self.n_queries,
            "mean_coherence": round(self.mean_coherence, 4),
            "mean_render_confidence": round(self.mean_render_confidence, 4),
            "mean_wave_confidence": round(self.mean_wave_confidence, 4),
            "mean_steps": round(self.mean_steps, 2),
            "termination_distribution": self.termination_distribution,
            "novelty_is_decaying": self.novelty_is_decaying,
        }
        if self.causal is not None:
            d["causal_score"] = round(self.causal.causal_score, 4)
        if self.locality is not None:
            d["locality_satisfied"] = self.locality.locality_satisfied
            d["max_distant_shift"] = self.locality.max_distant_shift
        d.update(self.extra)
        return d

    def __repr__(self) -> str:
        lines = [
            f"SuiteResult(",
            f"  n_queries         = {self.n_queries}",
            f"  mean_coherence    = {self.mean_coherence:.4f}",
            f"  mean_render_conf  = {self.mean_render_confidence:.4f}",
            f"  mean_wave_conf    = {self.mean_wave_confidence:.4f}",
            f"  mean_steps        = {self.mean_steps:.1f}",
            f"  terminations      = {self.termination_distribution}",
        ]
        if self.causal is not None:
            lines.append(f"  causal_score      = {self.causal.causal_score:.4f}")
        if self.locality is not None:
            lines.append(f"  locality_ok       = {self.locality.locality_satisfied}")
        if self.novelty_decay:
            lines.append(f"  novelty_decay     = {[round(x, 3) for x in self.novelty_decay]}")
        lines.append(")")
        return "\n".join(lines)
