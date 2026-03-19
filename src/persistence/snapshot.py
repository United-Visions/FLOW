"""ManifoldSnapshot — save / load the full Living Manifold state.

Serialises the entire M(t) as a single .npz file.  On load, M₀ is
re-derived via SeedGeometryEngine.build() (deterministic, <0.01s),
then the mutable state is restored from the saved arrays.

Arrays stored
-------------
labels          : object  (N,)      — concept labels
positions       : float64 (N, 104)  — current positions
densities       : float64 (N,)      — ρ(t) per point
deformations    : float64 (N, 104)  — accumulated φ(t)
curvatures      : float64 (N,)      — κ(t) per point
manifold_time   : float64 (1,)      — current t
n_writes        : uint64  (1,)      — write counter
format_version  : uint32  (1,)      — migration support
dimension       : uint32  (1,)      — always 104

What is NOT saved (rebuilt lazily on first query):
- KD-tree spatial index
- kNN geodesic graph
- Composer / geometry class instances
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

_FORMAT_VERSION = 1
_DIMENSION = 104


class ManifoldSnapshot:
    """Static utility for .npz manifold serialisation.

    All methods are class-level — no instantiation needed.
    Mirrors the VocabularyStore pattern from Phase 7.
    """

    @classmethod
    def save(cls, manifold, path: str) -> int:
        """Serialise a LivingManifold to a .npz file.

        Parameters
        ----------
        manifold : LivingManifold
            The manifold whose mutable state should be persisted.
        path : str
            File path; parent directories created if they do not exist.

        Returns
        -------
        int — number of points saved.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        labels_list = manifold.labels
        n = len(labels_list)
        dim = manifold.DIM

        # Build dense arrays from the manifold's internal stores
        positions = np.zeros((n, dim), dtype=np.float64)
        densities = np.zeros(n, dtype=np.float64)
        deformations = np.zeros((n, dim), dtype=np.float64)
        curvatures = np.zeros(n, dtype=np.float64)
        labels_arr = np.empty(n, dtype=object)

        for i, label in enumerate(labels_list):
            labels_arr[i] = label
            positions[i] = manifold.position(label)
            densities[i] = manifold._state.density.get(label)
            deformations[i] = manifold._state.deformation.displacement(label)
            curvatures[i] = manifold._state.get_curvature(label)

        np.savez_compressed(
            path,
            labels=labels_arr,
            positions=positions,
            densities=densities,
            deformations=deformations,
            curvatures=curvatures,
            manifold_time=np.array([manifold.t], dtype=np.float64),
            n_writes=np.array([manifold.n_writes], dtype=np.uint64),
            format_version=np.array([_FORMAT_VERSION], dtype=np.uint32),
            dimension=np.array([dim], dtype=np.uint32),
        )

        return n

    @classmethod
    def load(cls, path: str, manifold=None):
        """Load manifold state from a .npz file.

        If *manifold* is None a fresh LivingManifold is created from M₀
        via ``SeedGeometryEngine().build()``.  If a LivingManifold is
        provided it is used as the target (the caller wants to restore
        state into an existing instance).

        Parameters
        ----------
        path : str
            Path to a .npz file created by ``ManifoldSnapshot.save()``.
        manifold : LivingManifold | None
            Optionally pass an existing manifold to restore into.

        Returns
        -------
        LivingManifold — either freshly constructed or the provided
        instance with state restored.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifold snapshot not found: {path}")

        data = np.load(path, allow_pickle=True)

        # Validate format version
        fmt = int(data["format_version"][0])
        if fmt != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported snapshot format version {fmt} "
                f"(expected {_FORMAT_VERSION})"
            )

        dim = int(data["dimension"][0])
        if dim != _DIMENSION:
            raise ValueError(
                f"Snapshot dimension {dim} ≠ expected {_DIMENSION}"
            )

        labels = data["labels"]
        positions = data["positions"].astype(np.float64)
        densities = data["densities"].astype(np.float64)
        deformations = data["deformations"].astype(np.float64)
        curvatures = data["curvatures"].astype(np.float64)
        manifold_time = float(data["manifold_time"][0])
        n_writes = int(data["n_writes"][0])

        # Build or reuse the manifold
        if manifold is None:
            from src.phase1.seed_geometry.engine import SeedGeometryEngine
            from src.phase2.living_manifold.manifold import LivingManifold
            M0 = SeedGeometryEngine().build()
            manifold = LivingManifold(M0)

        # Restore all saved points (seed points are already there from
        # LivingManifold.__init__; non-seed points need to be added)
        seed_labels = set(manifold.labels)

        for i, label in enumerate(labels):
            label = str(label)
            vec = positions[i]
            if label not in seed_labels:
                # New point — place it on the manifold
                manifold.place(label, vec, origin="snapshot")
            else:
                # Existing seed point — update to its saved position
                manifold._points[label] = vec.copy()
                manifold._geodesic.update_point(label, vec)

            # Restore density + curvature + deformation
            manifold._state.density.set(label, float(densities[i]))
            manifold._state.set_curvature(label, float(curvatures[i]))
            manifold._state.deformation._displacements[label] = deformations[i].copy()

        # Restore time counters
        manifold._state.t = manifold_time
        manifold._state.n_writes = n_writes

        # Mark spatial indices as dirty so they rebuild on next query
        manifold._kdtree_dirty = True
        manifold._geodesic._dirty = True

        return manifold

    @classmethod
    def info(cls, path: str) -> dict:
        """Return metadata about a snapshot without loading all arrays.

        Returns
        -------
        dict with keys: n_points, dimension, format_version, manifold_time,
        n_writes.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifold snapshot not found: {path}")
        data = np.load(path, allow_pickle=True)
        return {
            "n_points": len(data["labels"]),
            "dimension": int(data["dimension"][0]),
            "format_version": int(data["format_version"][0]),
            "manifold_time": float(data["manifold_time"][0]),
            "n_writes": int(data["n_writes"][0]),
        }
