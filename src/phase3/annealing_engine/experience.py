"""Experience data types for the Annealing Engine.

An `Experience` is the unit of raw input that the Annealing Engine receives
from the outside world.  It carries a 104D vector representation and an
optional label.

An `ExperienceResult` records what happened when the engine processed a
single experience — the position it was placed at, the novelty score, the
applied temperature, the displacement magnitude, and how many manifold
points were affected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Experience:
    """A single raw experience to be ingested by the Annealing Engine.

    Attributes
    ----------
    vector : np.ndarray
        104D continuous representation of the experience.
    label  : str | None
        Optional human-readable label.  If provided and the engine is
        configured with ``place_labeled=True``, the experience will be
        inserted into the manifold under this label.
    source : str
        Textual tag describing where this experience came from
        (e.g. "sensory", "memory", "synthetic").  Never affects geometry.
    """

    vector: np.ndarray
    label: Optional[str] = None
    source: str = "raw"

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float)
        if self.vector.ndim != 1:
            raise ValueError(
                f"Experience.vector must be 1D, got shape {self.vector.shape}"
            )

    @property
    def dim(self) -> int:
        """Dimensionality of the experience vector."""
        return int(self.vector.shape[0])


@dataclass
class ExperienceResult:
    """Outcome of processing a single experience through the Annealing Engine.

    Attributes
    ----------
    experience      : the original Experience object
    located_label   : label of the nearest existing manifold point used as
                      the resonance anchor (None if manifold was empty)
    located_position: 104D position of the resonance anchor
    novelty         : novelty score in [0, 1]
    temperature     : temperature T(t) at processing time
    delta_magnitude : L2 norm of the applied displacement vector
    n_affected      : number of manifold points displaced by this experience
    placed_label    : label under which the experience was placed on the
                      manifold (None if ``place_labeled`` was False or no
                      label was provided)
    """

    experience: Experience
    located_label: Optional[str]
    located_position: np.ndarray
    novelty: float
    temperature: float
    delta_magnitude: float
    n_affected: int
    placed_label: Optional[str] = None

    @property
    def was_novel(self) -> bool:
        """True when novelty > 0.5 — experience was genuinely new."""
        return self.novelty > 0.5

    @property
    def deformation_applied(self) -> bool:
        """True when the deformation was non-negligible."""
        return self.delta_magnitude > 1e-9

    def __repr__(self) -> str:  # pragma: no cover
        label_str = self.experience.label or "<unlabeled>"
        return (
            f"ExperienceResult("
            f"label={label_str!r}, "
            f"novelty={self.novelty:.3f}, "
            f"T={self.temperature:.4f}, "
            f"|δ|={self.delta_magnitude:.4f}, "
            f"n_affected={self.n_affected})"
        )
