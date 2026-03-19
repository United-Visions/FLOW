"""PipelineResult — output data type for a single GEOPipeline query.

A PipelineResult bundles the intermediate artefacts from every component
so that callers can inspect the full reasoning chain:

  C5  →  trajectory   (the continuous path through M)
  C6  →  wave         (the pre-linguistic standing wave Ψ)
  C7  →  output       (the rendered natural-language text)

This type has NO access to the manifold.  It is a value object.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.phase4.flow_engine.query import Query, Trajectory
from src.phase1.expression.wave import StandingWave
from src.phase1.expression.renderer import RenderedOutput


@dataclass
class PipelineResult:
    """Bundle of all artefacts produced by a single end-to-end query.

    Attributes
    ----------
    query               : the original Query submitted to the pipeline
    trajectory          : the continuous trajectory produced by C5
    wave                : the standing wave Ψ produced by C6
    output              : the RenderedOutput produced by C7

    Derived properties
    ------------------
    text                : rendered natural-language text (str)
    confidence          : rendering confidence from C7 (float in [0,1])
    n_steps             : number of SDE integration steps in the trajectory
    termination_reason  : why the flow stopped
    wave_confidence     : wave confidence from C6 (float in [0,1])
    mean_speed          : mean flow speed (float)
    mean_curvature      : mean trajectory curvature (float)
    """

    query: Query
    trajectory: Trajectory
    wave: StandingWave
    output: RenderedOutput

    # ── Derived convenience properties ────────────────────────────────────

    @property
    def text(self) -> str:
        """Rendered natural-language text from C7."""
        return self.output.text

    @property
    def confidence(self) -> float:
        """Rendering confidence from C7 ( avg resonance score ∈ [0,1] )."""
        return self.output.confidence

    @property
    def n_steps(self) -> int:
        """Number of SDE integration steps in the flow trajectory."""
        return len(self.trajectory)

    @property
    def termination_reason(self) -> str:
        """Why the flow terminated (velocity_threshold / revisit_detected /
        max_steps / attractor_reached)."""
        return self.trajectory.termination_reason

    @property
    def wave_confidence(self) -> float:
        """Standing-wave confidence from C6 (overall wave quality)."""
        return self.wave.mean_confidence()

    @property
    def mean_speed(self) -> float:
        """Mean speed of the flow trajectory."""
        return self.trajectory.mean_speed

    @property
    def mean_curvature(self) -> float:
        """Mean curvature of the flow trajectory."""
        return self.trajectory.mean_curvature

    @property
    def flow_preserved(self) -> bool:
        """Whether C7 successfully preserved the flow dynamics."""
        return self.output.flow_preserved

    def __repr__(self) -> str:
        return (
            f"PipelineResult(\n"
            f"  query={self.query.label!r}\n"
            f"  n_steps={self.n_steps}, "
            f"reason={self.termination_reason!r}\n"
            f"  wave_confidence={self.wave_confidence:.3f}, "
            f"render_confidence={self.confidence:.3f}\n"
            f"  text={self.text[:80]!r}{'...' if len(self.text) > 80 else ''}\n"
            f")"
        )
