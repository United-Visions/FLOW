"""Region classification for manifold points.

Three region types emerge from local density:

  CRYSTALLIZED  — high density, stiff geometry, confident
  FLEXIBLE      — medium density, still deforming, moderate confidence
  UNKNOWN       — low density, highly deformable, low confidence
"""

from __future__ import annotations

from enum import Enum


class RegionType(Enum):
    """Qualitative region type based on local density."""

    CRYSTALLIZED = "crystallized"   # high density  (ρ > HIGH_THRESHOLD)
    FLEXIBLE = "flexible"           # medium density (LOW_THRESHOLD < ρ ≤ HIGH_THRESHOLD)
    UNKNOWN = "unknown"             # low density   (ρ ≤ LOW_THRESHOLD)


class RegionClassifier:
    """Map a density scalar to a region type and derived properties.

    Parameters
    ----------
    high_threshold : float
        Density above which a region is crystallised.  Default 0.60.
    low_threshold : float
        Density below which a region is unknown territory.  Default 0.20.
    """

    HIGH_THRESHOLD: float = 0.60
    LOW_THRESHOLD: float = 0.20

    def __init__(
        self,
        high_threshold: float = HIGH_THRESHOLD,
        low_threshold: float = LOW_THRESHOLD,
    ) -> None:
        if not (0.0 < low_threshold < high_threshold <= 1.0):
            raise ValueError(
                f"Require 0 < low_threshold({low_threshold}) < "
                f"high_threshold({high_threshold}) ≤ 1"
            )
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #

    def classify(self, density: float) -> RegionType:
        """Return the *RegionType* for the given *density* scalar."""
        if density > self.high_threshold:
            return RegionType.CRYSTALLIZED
        if density > self.low_threshold:
            return RegionType.FLEXIBLE
        return RegionType.UNKNOWN

    # ------------------------------------------------------------------ #
    # Derived physical properties                                          #
    # ------------------------------------------------------------------ #

    def stiffness(self, density: float) -> float:
        """Geometric stiffness in [0, 1].

        Stiffness = resistance to deformation.
        Crystallised regions are nearly rigid (stiffness ≈ 1).
        Unknown territory is fully fluid (stiffness ≈ 0).
        """
        return float(density)  # density IS the normalised stiffness

    def flexibility(self, density: float) -> float:
        """Complement of stiffness: how easily the region deforms."""
        return 1.0 - self.stiffness(density)

    def locality_radius(self, density: float, r_max: float = 5.0) -> float:
        """Deformation locality radius as a function of density.

        Denser (stiffer) regions have *smaller* locality radius — changes
        are confined to a tighter neighbourhood.  Sparse regions allow
        wider propagation.

        r = r_max · exp(−density · 3)
        """
        import math

        return r_max * math.exp(-density * 3.0)

    def diffusion_scale(self, density: float, base: float = 1.0) -> float:
        """SDE diffusion scale σ — larger in flexible / unknown regions."""
        return base * self.flexibility(density)

    def confidence_from_density(self, density: float) -> float:
        """Heuristic confidence score derived from density.

        Crystallised regions ≈ 1.0; unknown territory ≈ 0.0.
        """
        return float(density)

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #

    def describe(self, density: float) -> str:
        region = self.classify(density)
        return (
            f"{region.value}  "
            f"(density={density:.3f}, "
            f"stiffness={self.stiffness(density):.3f}, "
            f"r_locality={self.locality_radius(density):.3f})"
        )
