"""Simplified persistent homology tracker for the Contrast Engine.

Persistent homology tracks topological features (connected components,
loops, voids) over time by observing how pairwise distances between
concepts evolve.  Features that survive for a long time are real structure;
features that die quickly are noise.

This implementation provides a practical approximation:
  - Tracks pairwise distance records over time
  - Detects 0-dimensional persistence (connected components / clusters)
    using a union-find structure at various distance thresholds
  - Identifies persistent clusters as structural corrections

Per the spec, corrections are fed back to the Living Manifold as
deformation suggestions — not as labels or explicit cluster assignments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PersistenceEvent:
    """A topological feature birth or death event.

    Attributes
    ----------
    label_a, label_b : labels of the two concepts involved
    distance_at_birth : pairwise distance when the feature appeared
    distance_at_death : pairwise distance when the feature disappeared
                        (inf if still alive)
    birth_time        : manifold time when feature was born
    death_time        : manifold time at death (inf if still alive)
    """

    label_a: str
    label_b: str
    distance_at_birth: float
    distance_at_death: float = math.inf
    birth_time: float = 0.0
    death_time: float = math.inf

    @property
    def lifetime(self) -> float:
        """persistence = |death - birth| in distance space."""
        return abs(self.distance_at_death - self.distance_at_birth)

    @property
    def is_alive(self) -> bool:
        return math.isinf(self.distance_at_death)


@dataclass
class DistanceRecord:
    """Timestamped pairwise distance between two concepts."""

    label_a: str
    label_b: str
    distance: float
    time: float


class PersistenceDiagram:
    """Track pairwise distances over time and detect persistent features.

    Usage:
      diagram = PersistenceDiagram()
      diagram.record("cat", "dog", dist, t)   # after each contrast op
      corrections = diagram.cluster_corrections(min_lifetime=5.0)
    """

    def __init__(self, cluster_threshold_init: float = 2.0) -> None:
        """
        Parameters
        ----------
        cluster_threshold_init : float
            Initial distance threshold below which two concepts are
            considered in the same connected component.
        """
        self.cluster_threshold = cluster_threshold_init
        # (label_a, label_b) → list of DistanceRecord, sorted by time
        self._history: Dict[Tuple[str, str], List[DistanceRecord]] = {}
        # Active persistence events
        self._events: List[PersistenceEvent] = []

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record(
        self, label_a: str, label_b: str, distance: float, time: float
    ) -> None:
        """Record a pairwise distance observation."""
        key = self._key(label_a, label_b)
        rec = DistanceRecord(label_a=label_a, label_b=label_b, distance=distance, time=time)
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(rec)
        # Update active events (simplified: one event per pair)
        self._update_events(label_a, label_b, distance, time)

    def _update_events(
        self, label_a: str, label_b: str, distance: float, time: float
    ) -> None:
        """Maintain birth/death events for each pair."""
        key = self._key(label_a, label_b)
        # Find existing live event for this pair
        live: Optional[PersistenceEvent] = None
        for ev in self._events:
            if self._key(ev.label_a, ev.label_b) == key and ev.is_alive:
                live = ev
                break

        currently_close = distance <= self.cluster_threshold

        if live is None:
            if currently_close:
                # Birth
                self._events.append(
                    PersistenceEvent(
                        label_a=label_a,
                        label_b=label_b,
                        distance_at_birth=distance,
                        birth_time=time,
                    )
                )
        else:
            if not currently_close:
                # Death — they drifted apart
                live.distance_at_death = distance
                live.death_time = time

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_persistent_features(
        self, min_lifetime: float = 2.0
    ) -> List[PersistenceEvent]:
        """Return events whose persistence (lifetime) exceeds *min_lifetime*.

        Both alive and dead events are included if they meet the threshold.
        """
        return [ev for ev in self._events if ev.lifetime >= min_lifetime]

    def current_distances(self) -> Dict[Tuple[str, str], float]:
        """Return the most recently recorded distance for each pair."""
        result: Dict[Tuple[str, str], float] = {}
        for key, records in self._history.items():
            if records:
                result[key] = records[-1].distance
        return result

    def cluster_corrections(
        self, min_lifetime: float = 2.0
    ) -> List[Dict]:
        """Suggest deformation corrections based on persistent clusters.

        Each correction is a dict with:
          type        : "tighten" or "separate"
          label_a     : first concept
          label_b     : second concept
          strength    : float in [0,1] — how strongly to apply
          reasoning   : human-readable explanation
        """
        corrections = []
        persistent = self.get_persistent_features(min_lifetime)

        for ev in persistent:
            if ev.is_alive:
                # Long-lived closeness → should remain tight
                strength = min(ev.lifetime / 10.0, 1.0)
                corrections.append(
                    {
                        "type": "tighten",
                        "label_a": ev.label_a,
                        "label_b": ev.label_b,
                        "strength": strength,
                        "reasoning": (
                            f"Concepts '{ev.label_a}' and '{ev.label_b}' "
                            f"have been close for lifetime={ev.lifetime:.2f} — "
                            "reinforcing cluster."
                        ),
                    }
                )
            else:
                # Feature died — they separated; reinforce the separation
                strength = min(ev.lifetime / 10.0, 1.0)
                corrections.append(
                    {
                        "type": "separate",
                        "label_a": ev.label_a,
                        "label_b": ev.label_b,
                        "strength": strength,
                        "reasoning": (
                            f"Concepts '{ev.label_a}' and '{ev.label_b}' "
                            f"separated after lifetime={ev.lifetime:.2f} — "
                            "reinforcing boundary."
                        ),
                    }
                )
        return corrections

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _key(a: str, b: str) -> Tuple[str, str]:
        """Canonical (sorted) key for a pair, so (a,b) == (b,a)."""
        return (min(a, b), max(a, b))

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        n_alive = sum(1 for ev in self._events if ev.is_alive)
        return (
            f"PersistenceDiagram(pairs={len(self)}, events={len(self._events)}, "
            f"alive={n_alive})"
        )
