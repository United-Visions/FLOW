"""Local deformation operator for the Living Manifold.

Implements the locality guarantee:

  effect(Q) → 0  as  distance(P, Q) → ∞

Deformations are applied using a Gaussian kernel centred at the target
point P.  Only points within 3σ of P are affected, where σ equals the
locality radius of P (which shrinks in dense / crystallised regions).

The stiffness of each affected point Q further scales the displacement:

  Δq = w(P, Q) · stiffness_factor(Q) · δ

where w is the Gaussian weight and stiffness_factor = (1 - density(Q)).
This means well-established concepts resist deformation naturally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class DeformationResult:
    """Output of a single deform_local call.

    Attributes
    ----------
    centre_label   : label of the point at which the deformation is applied
    affected       : list of (label, displacement_vector) for every moved point
    locality_radius: the radius used for this deformation
    n_affected     : number of points affected
    """

    centre_label: str
    affected: List[Tuple[str, np.ndarray]]
    locality_radius: float
    n_affected: int


class LocalDeformation:
    """Apply a Gaussian-weighted local deformation to a set of points.

    Parameters
    ----------
    cutoff_sigma : float
        Points beyond cutoff_sigma × locality_radius are unaffected.
        Default 3.0.
    """

    def __init__(self, cutoff_sigma: float = 3.0) -> None:
        self.cutoff_sigma = cutoff_sigma

    # ------------------------------------------------------------------ #
    # Main interface                                                       #
    # ------------------------------------------------------------------ #

    def apply(
        self,
        centre_label: str,
        centre_vector: np.ndarray,
        delta: np.ndarray,
        locality_radius: float,
        all_points: Dict[str, np.ndarray],
        density_func,              # callable(label) → float in [0,1]
        candidate_labels: Optional[Set[str]] = None,
    ) -> DeformationResult:
        """Compute displacements for all affected points.

        Parameters
        ----------
        centre_label    : label of the deformation target
        centre_vector   : current 104D position of the centre
        delta           : desired displacement at the centre (104D)
        locality_radius : radius σ — derived from density of centre
        all_points      : {label: vector} for all points on the manifold
        density_func    : callable returning density in [0,1] for a label
        candidate_labels: optional pre-filtered set of labels within
                          the cutoff radius (e.g. from cKDTree range query).
                          When provided, only these labels are scanned
                          instead of all_points — O(k_local) instead of O(n).

        Returns
        -------
        DeformationResult with displacements for every affected point.
        """
        cutoff = self.cutoff_sigma * locality_radius
        affected: List[Tuple[str, np.ndarray]] = []

        # If candidates are pre-filtered, iterate only those;
        # always include the centre itself.
        if candidate_labels is not None:
            scan_labels = candidate_labels | {centre_label}
        else:
            scan_labels = set(all_points.keys())

        for label in scan_labels:
            vec = all_points.get(label)
            if vec is None:
                continue
            dist = float(np.linalg.norm(vec - centre_vector))
            if dist > cutoff:
                continue

            # Gaussian weight
            if locality_radius < 1e-12:
                w = 1.0 if label == centre_label else 0.0
            else:
                w = math.exp(-(dist ** 2) / (2.0 * locality_radius ** 2))

            # Stiffness factor: stiffer points resist neighbourhood drag.
            # The centre point is always moved at full weight — crystallisation
            # means neighbours are harder to drag, not that the target itself
            # cannot be deliberately displaced.
            density = density_func(label)
            resistance = 0.0 if label == centre_label else density
            effective_w = w * (1.0 - resistance)

            if effective_w < 1e-9:
                continue

            affected.append((label, effective_w * delta))

        return DeformationResult(
            centre_label=centre_label,
            affected=affected,
            locality_radius=locality_radius,
            n_affected=len(affected),
        )

    # ------------------------------------------------------------------ #
    # Locality validation                                                  #
    # ------------------------------------------------------------------ #

    def validate_locality(
        self,
        result: DeformationResult,
        all_vectors: Dict[str, np.ndarray],
        centre_vector: np.ndarray,
        max_radius: float,
    ) -> bool:
        """Return True if no displacement exceeds the expected locality radius.

        This is the hard constraint from the spec:
          effect(Q) → 0 as distance(P,Q) → ∞
        We check the weaker version: no affected point is further than
        cutoff_sigma × locality_radius from the centre.
        """
        cutoff = self.cutoff_sigma * result.locality_radius

        # Allow a small tolerance above max_radius
        allowed = max(cutoff, max_radius) * 1.05

        for label, displacement in result.affected:
            if label not in all_vectors:
                continue
            dist = float(
                np.linalg.norm(all_vectors[label] - centre_vector)
            )
            # If displacement is negligible, ignore
            if np.linalg.norm(displacement) < 1e-12:
                continue
            if dist > allowed:
                return False
        return True
