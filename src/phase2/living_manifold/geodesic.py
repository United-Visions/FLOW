"""Approximate geodesic computation on the Living Manifold.

At Phase 2 scale (thousands of points) we approximate geodesics as
shortest paths through a kNN neighbourhood graph.  Euclidean distances
in the 104-dimensional bundle space are used as edge weights —
this is a first-order approximation of the true Riemannian geodesic.

The approximation quality improves as point density increases; at
very low density the graph may be disconnected, in which case we fall
back to the straight-line Euclidean distance.

Phase 8 upgrade: incremental graph maintenance
-----------------------------------------------
Instead of rebuilding the full O(n²) graph on every mutation, we
track a set of dirty labels and only recompute edges for those
points plus their immediate kNN neighbours.  Full rebuild occurs
only on first construction or when the dirty fraction exceeds a
threshold.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class GeodesicComputer:
    """Compute approximate geodesics through a kNN graph.

    Parameters
    ----------
    k_neighbours : int
        Number of nearest neighbours each point is connected to in the graph.
    rebuild_fraction : float
        If the fraction of dirty labels exceeds this threshold, do a full
        rebuild instead of incremental updates.  Default 0.3.
    """

    def __init__(self, k_neighbours: int = 8, rebuild_fraction: float = 0.3) -> None:
        self.k_neighbours = k_neighbours
        self.rebuild_fraction = rebuild_fraction
        # label → 104D vector
        self._vectors: Dict[str, np.ndarray] = {}
        # label → list of (neighbour_label, edge_weight)
        self._graph: Dict[str, List[Tuple[str, float]]] = {}
        self._dirty: bool = False  # graph needs rebuild
        self._dirty_labels: Set[str] = set()  # labels needing edge refresh
        self._fully_built: bool = False  # has the graph ever been fully built?

    # ------------------------------------------------------------------ #
    # Graph maintenance                                                    #
    # ------------------------------------------------------------------ #

    def add_point(self, label: str, vector: np.ndarray) -> None:
        """Register a point; marks the label as needing edge refresh."""
        self._vectors[label] = vector.copy()
        self._dirty_labels.add(label)
        self._dirty = True

    def remove_point(self, label: str) -> None:
        """Remove a point and mark the graph as needing rebuild."""
        self._vectors.pop(label, None)
        self._graph.pop(label, None)
        self._dirty_labels.discard(label)
        # Remove from all neighbour lists
        for neighbours in self._graph.values():
            neighbours[:] = [(nb, w) for nb, w in neighbours if nb != label]
        self._dirty = True

    def update_point(self, label: str, vector: np.ndarray) -> None:
        """Update position of an existing point (e.g. after deformation)."""
        self._vectors[label] = vector.copy()
        self._dirty_labels.add(label)
        self._dirty = True

    def _ensure_graph(self) -> None:
        """Ensure the graph is up to date — full or incremental rebuild."""
        if not self._dirty:
            return
        n = len(self._vectors)
        if n == 0:
            self._graph = {}
            self._dirty = False
            self._dirty_labels.clear()
            self._fully_built = True
            return

        # Decide: full rebuild or incremental
        dirty_fraction = len(self._dirty_labels) / max(n, 1)
        if not self._fully_built or dirty_fraction > self.rebuild_fraction:
            self._rebuild_graph()
        else:
            self._incremental_update()

    def _rebuild_graph(self) -> None:
        """Rebuild the kNN adjacency graph from current vectors."""
        labels = list(self._vectors.keys())
        n = len(labels)
        if n == 0:
            self._graph = {}
            self._dirty = False
            self._dirty_labels.clear()
            self._fully_built = True
            return

        # Stack all vectors
        matrix = np.stack([self._vectors[l] for l in labels])  # (n, dim)

        self._graph = {l: [] for l in labels}

        k = min(self.k_neighbours, n - 1)
        if k == 0:
            self._dirty = False
            self._dirty_labels.clear()
            self._fully_built = True
            return

        for i, li in enumerate(labels):
            vi = matrix[i]
            # Squared Euclidean distances to all other points
            diff = matrix - vi  # (n, dim)
            sq_dists = np.sum(diff ** 2, axis=1)
            sq_dists[i] = np.inf  # exclude self

            # k nearest indices
            nn_idx = np.argpartition(sq_dists, k)[:k]
            for j in nn_idx:
                lj = labels[j]
                w = math.sqrt(float(sq_dists[j]))
                self._graph[li].append((lj, w))
                self._graph[lj].append((li, w))  # undirected

        # De-duplicate edges
        for l in labels:
            seen: dict[str, float] = {}
            for (nb, w) in self._graph[l]:
                if nb not in seen or seen[nb] > w:
                    seen[nb] = w
            self._graph[l] = list(seen.items())

        self._dirty = False
        self._dirty_labels.clear()
        self._fully_built = True

    def _incremental_update(self) -> None:
        """Recompute edges only for dirty labels + their neighbours.

        Cost: O(d × n) per dirty label where d = dirty count and n = total.
        Much cheaper than full O(n²) when d << n.
        """
        if not self._dirty_labels:
            self._dirty = False
            return

        labels = list(self._vectors.keys())
        n = len(labels)
        k = min(self.k_neighbours, n - 1)
        if k == 0:
            self._dirty = False
            self._dirty_labels.clear()
            return

        label_to_idx = {l: i for i, l in enumerate(labels)}
        matrix = np.stack([self._vectors[l] for l in labels])

        # Collect all labels that need edge refresh: dirty + their old neighbours
        refresh_set = set(self._dirty_labels)
        for lbl in self._dirty_labels:
            if lbl in self._graph:
                for nb, _ in self._graph[lbl]:
                    refresh_set.add(nb)

        for lbl in refresh_set:
            if lbl not in label_to_idx:
                continue
            i = label_to_idx[lbl]
            vi = matrix[i]

            # Compute distances to all points
            diff = matrix - vi
            sq_dists = np.sum(diff ** 2, axis=1)
            sq_dists[i] = np.inf

            nn_idx = np.argpartition(sq_dists, k)[:k]

            # Clear old edges from this node
            old_neighbours = [nb for nb, _ in self._graph.get(lbl, [])]
            self._graph[lbl] = []

            # Remove this node from old neighbours' lists
            for nb in old_neighbours:
                if nb in self._graph:
                    self._graph[nb] = [
                        (n2, w2) for n2, w2 in self._graph[nb] if n2 != lbl
                    ]

            # Add fresh edges
            for j in nn_idx:
                lj = labels[j]
                w = math.sqrt(float(sq_dists[j]))
                self._graph[lbl].append((lj, w))
                # Ensure the reverse edge exists
                if lbl not in {n2 for n2, _ in self._graph.get(lj, [])}:
                    if lj not in self._graph:
                        self._graph[lj] = []
                    self._graph[lj].append((lbl, w))

        # De-duplicate edges in touched nodes
        all_touched = refresh_set.copy()
        for lbl in refresh_set:
            for nb, _ in self._graph.get(lbl, []):
                all_touched.add(nb)

        for lbl in all_touched:
            if lbl not in self._graph:
                continue
            seen: dict[str, float] = {}
            for nb, w in self._graph[lbl]:
                if nb not in seen or seen[nb] > w:
                    seen[nb] = w
            self._graph[lbl] = list(seen.items())

        self._dirty = False
        self._dirty_labels.clear()

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def distance(self, label_a: str, label_b: str) -> float:
        """Approximate geodesic distance between two labelled points.

        Falls back to Euclidean distance if the graph is disconnected.
        """
        if label_a == label_b:
            return 0.0
        self._ensure_graph()
        dist, _ = self._dijkstra(label_a, label_b)
        return dist

    def path(self, label_a: str, label_b: str) -> List[str]:
        """Return ordered list of labels on the approximate geodesic path."""
        if label_a == label_b:
            return [label_a]
        self._ensure_graph()
        _, path = self._dijkstra(label_a, label_b)
        return path

    def _dijkstra(
        self, source: str, target: str
    ) -> Tuple[float, List[str]]:
        """Dijkstra's algorithm with path reconstruction."""
        if source not in self._graph or target not in self._graph:
            # Fall back to Euclidean
            if source in self._vectors and target in self._vectors:
                d = float(np.linalg.norm(self._vectors[source] - self._vectors[target]))
                return d, [source, target]
            return float("inf"), []

        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}
        heap: List[Tuple[float, str]] = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if u == target:
                break
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self._graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if target not in dist:
            # Disconnected — fall back to Euclidean
            if source in self._vectors and target in self._vectors:
                d = float(np.linalg.norm(self._vectors[source] - self._vectors[target]))
                return d, [source, target]
            return float("inf"), []

        # Reconstruct path
        path: List[str] = []
        cur: Optional[str] = target
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return dist[target], path

    # ------------------------------------------------------------------ #
    # Bulk                                                                 #
    # ------------------------------------------------------------------ #

    def all_distances_from(self, source: str) -> Dict[str, float]:
        """Dijkstra from *source* to all reachable points."""
        self._ensure_graph()
        if source not in self._graph:
            return {}

        dist: Dict[str, float] = {source: 0.0}
        heap: List[Tuple[float, str]] = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self._graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def __len__(self) -> int:
        return len(self._vectors)
