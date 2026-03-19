"""Mutable state containers for the Living Manifold.

M(t) = (M₀, φ(t), ρ(t), κ(t))
  M₀   = seed geometry (fixed)
  φ(t) = deformation field
  ρ(t) = density function
  κ(t) = curvature tensor (scalar approximation per point)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class DeformationField:
    """Accumulated displacement vectors φ(t) for each labelled point.

    Each entry maps a point label to the total displacement accumulated
    by calls to deform_local().  The displacement is stored in the same
    104-dimensional space as the manifold coordinates.
    """

    _displacements: Dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Mutation                                                             #
    # ------------------------------------------------------------------ #

    def accumulate(self, label: str, displacement: np.ndarray) -> None:
        """Add *displacement* to the running total for *label*."""
        if label in self._displacements:
            self._displacements[label] = self._displacements[label] + displacement
        else:
            self._displacements[label] = displacement.copy()

    def register(self, label: str, dim: int) -> None:
        """Ensure *label* has a zero-displacement entry (idempotent)."""
        if label not in self._displacements:
            self._displacements[label] = np.zeros(dim)

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    def displacement(self, label: str) -> np.ndarray:
        """Return accumulated displacement for *label*, or zero array."""
        return self._displacements.get(label, np.zeros(1))

    def has(self, label: str) -> bool:
        return label in self._displacements

    def labels(self) -> list[str]:
        return list(self._displacements.keys())

    def __len__(self) -> int:
        return len(self._displacements)


@dataclass
class DensityField:
    """Local density ρ(t) at each labelled point.

    Density is a normalised scalar in [0, 1].
    0 = empty / unknown territory.
    1 = maximally crystallised.
    """

    _density: Dict[str, float] = field(default_factory=dict)

    def set(self, label: str, value: float) -> None:
        self._density[label] = float(np.clip(value, 0.0, 1.0))

    def get(self, label: str) -> float:
        return self._density.get(label, 0.0)

    def labels(self) -> list[str]:
        return list(self._density.keys())

    def __len__(self) -> int:
        return len(self._density)


@dataclass
class ManifoldState:
    """Complete mutable state of the Living Manifold at time t.

    Attributes
    ----------
    deformation : DeformationField
        Accumulated deformation φ(t) per point.
    density     : DensityField
        Local density ρ(t) per point.
    curvature   : dict[str, float]
        Scalar curvature approximation κ(t) per point.
    t           : float
        Manifold time (number of write operations / wall-clock blend).
    n_writes    : int
        Total write operations committed so far.
    """

    deformation: DeformationField = field(default_factory=DeformationField)
    density: DensityField = field(default_factory=DensityField)
    curvature: Dict[str, float] = field(default_factory=dict)
    t: float = 0.0
    n_writes: int = 0
    _wall_start: float = field(default_factory=time.monotonic, repr=False)

    # ------------------------------------------------------------------ #
    # Curvature helpers                                                    #
    # ------------------------------------------------------------------ #

    def set_curvature(self, label: str, value: float) -> None:
        self.curvature[label] = float(value)

    def get_curvature(self, label: str) -> float:
        return self.curvature.get(label, 0.0)

    # ------------------------------------------------------------------ #
    # Time progression                                                     #
    # ------------------------------------------------------------------ #

    def tick(self) -> None:
        """Advance manifold time by one write operation."""
        self.n_writes += 1
        self.t = self.n_writes + (time.monotonic() - self._wall_start) * 0.001
