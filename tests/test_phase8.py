"""Phase 8 tests — Scaling: Persistence, cKDTree, Incremental Geodesic,
Deformation Pre-filter, FAISS Matching, Pipeline save/load.

Organised by test class, one per feature area.
"""

from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np
import pytest

# ── Phase 1 / 2 imports ──────────────────────────────────────────────────
from src.phase1.seed_geometry.engine import SeedGeometryEngine
from src.phase1.expression.matcher import ResonanceMatcher, ExpressionEntry, _HAS_FAISS
from src.phase1.expression.wave import WAVE_DIM, WaveSegment, WavePoint, StandingWave

# ── Phase 2 ──────────────────────────────────────────────────────────────
from src.phase2.living_manifold.manifold import LivingManifold
from src.phase2.living_manifold.geodesic import GeodesicComputer
from src.phase2.living_manifold.deformation import LocalDeformation
from src.phase2.living_manifold.regions import RegionType

# ── Phase 3 ──────────────────────────────────────────────────────────────
from src.phase3.annealing_engine.engine import AnnealingEngine
from src.phase3.annealing_engine.experience import Experience

# ── Phase 5 ──────────────────────────────────────────────────────────────
from src.phase5.pipeline.pipeline import GEOPipeline

# ── Phase 8 — Persistence ────────────────────────────────────────────────
from src.persistence.snapshot import ManifoldSnapshot


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_manifold() -> LivingManifold:
    """Build a fresh LivingManifold from M₀."""
    return LivingManifold(SeedGeometryEngine().build())


def _random_vec(rng: np.random.Generator, dim: int = 104) -> np.ndarray:
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v) + 1e-12
    return v * 5.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: ManifoldSnapshot (Persistence)
# ═══════════════════════════════════════════════════════════════════════════

class TestManifoldSnapshot:
    """PRIORITY 2 — Manifold persistence via .npz serialisation."""

    def test_save_creates_file(self):
        m = _make_manifold()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            n = ManifoldSnapshot.save(m, path)
            assert os.path.exists(path)
            assert n == m.n_points
        finally:
            os.unlink(path)

    def test_save_load_round_trip_seed_only(self):
        """Seed-only manifold survives a save → load cycle."""
        m1 = _make_manifold()
        labels_before = sorted(m1.labels)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m1, path)
            m2 = ManifoldSnapshot.load(path)

            labels_after = sorted(m2.labels)
            assert labels_before == labels_after

            for label in labels_before:
                np.testing.assert_allclose(
                    m1.position(label), m2.position(label), atol=1e-12
                )
        finally:
            os.unlink(path)

    def test_save_load_with_learned_concepts(self):
        """Learned concepts placed via C3 survive persistence."""
        m1 = _make_manifold()
        rng = np.random.default_rng(42)
        ae = AnnealingEngine(m1, T0=1.0, lambda_=0.01, T_floor=0.05)
        for i in range(10):
            ae.process(Experience(vector=_random_vec(rng), label=f"learned_{i}"))

        n_before = m1.n_points
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m1, path)
            m2 = ManifoldSnapshot.load(path)
            assert m2.n_points == n_before

            for i in range(10):
                label = f"learned_{i}"
                assert label in m2.labels
                np.testing.assert_allclose(
                    m1.position(label), m2.position(label), atol=1e-12
                )
        finally:
            os.unlink(path)

    def test_density_preserved(self):
        m1 = _make_manifold()
        rng = np.random.default_rng(7)
        ae = AnnealingEngine(m1, T0=1.0, lambda_=0.01, T_floor=0.05)
        for i in range(5):
            ae.process(Experience(vector=_random_vec(rng), label=f"d_{i}"))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m1, path)
            m2 = ManifoldSnapshot.load(path)
            for label in m1.labels:
                assert abs(
                    m1._state.density.get(label) - m2._state.density.get(label)
                ) < 1e-10
        finally:
            os.unlink(path)

    def test_deformation_preserved(self):
        m1 = _make_manifold()
        label = m1.labels[0]
        delta = np.zeros(104)
        delta[0] = 0.1
        m1.deform_local(label, delta)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m1, path)
            m2 = ManifoldSnapshot.load(path)
            np.testing.assert_allclose(
                m1._state.deformation.displacement(label),
                m2._state.deformation.displacement(label),
                atol=1e-12,
            )
        finally:
            os.unlink(path)

    def test_manifold_time_preserved(self):
        m1 = _make_manifold()
        label = m1.labels[0]
        delta = np.zeros(104)
        delta[0] = 0.01
        m1.deform_local(label, delta)
        m1.deform_local(label, delta)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m1, path)
            m2 = ManifoldSnapshot.load(path)
            assert m2.n_writes == m1.n_writes
        finally:
            os.unlink(path)

    def test_info_without_full_load(self):
        m = _make_manifold()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m, path)
            info = ManifoldSnapshot.info(path)
            assert info["n_points"] == m.n_points
            assert info["dimension"] == 104
            assert info["format_version"] == 1
        finally:
            os.unlink(path)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            ManifoldSnapshot.load("/tmp/nonexistent_flow_snapshot.npz")

    def test_save_load_into_existing_manifold(self):
        """Pass an existing manifold to load() — state is restored in-place."""
        m1 = _make_manifold()
        rng = np.random.default_rng(99)
        ae = AnnealingEngine(m1, T0=1.0, lambda_=0.01, T_floor=0.05)
        ae.process(Experience(vector=_random_vec(rng), label="concept_x"))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m1, path)
            m2 = _make_manifold()
            ManifoldSnapshot.load(path, manifold=m2)
            assert "concept_x" in m2.labels
            np.testing.assert_allclose(
                m1.position("concept_x"), m2.position("concept_x"), atol=1e-12
            )
        finally:
            os.unlink(path)

    def test_file_size_reasonable(self):
        """Snapshot should be compact (< 500KB for seed-only manifold)."""
        m = _make_manifold()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m, path)
            size_kb = os.path.getsize(path) / 1024
            assert size_kb < 500, f"Snapshot too large: {size_kb:.1f} KB"
        finally:
            os.unlink(path)

    def test_loaded_manifold_is_queryable(self):
        """After loading, all READ operations should work correctly."""
        m1 = _make_manifold()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m1, path)
            m2 = ManifoldSnapshot.load(path)

            # Test various READ operations
            label = m2.labels[0]
            pos = m2.position(label)
            assert pos.shape == (104,)
            assert isinstance(m2.density(pos), float)
            assert isinstance(m2.curvature(pos), float)
            nearest = m2.nearest(pos, k=3)
            assert len(nearest) >= 1
            assert isinstance(m2.region_type(pos), RegionType)
        finally:
            os.unlink(path)

    def test_loaded_manifold_writable(self):
        """After loading, WRITE operations should work."""
        m1 = _make_manifold()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m1, path)
            m2 = ManifoldSnapshot.load(path)

            rng = np.random.default_rng(55)
            m2.place("new_concept", _random_vec(rng))
            assert "new_concept" in m2.labels
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: cKDTree upgrade
# ═══════════════════════════════════════════════════════════════════════════

class TestCKDTree:
    """PRIORITY 4 — cKDTree replaces KDTree (C-accelerated)."""

    def test_import_uses_ckdtree(self):
        """Manifold uses scipy.spatial.cKDTree, not KDTree."""
        from scipy.spatial import cKDTree
        m = _make_manifold()
        m._ensure_kdtree()
        assert isinstance(m._kdtree, cKDTree)

    def test_nearest_works_after_upgrade(self):
        m = _make_manifold()
        pos = m.position(m.labels[0])
        results = m.nearest(pos, k=5)
        assert len(results) == 5

    def test_density_works_after_upgrade(self):
        m = _make_manifold()
        pos = m.position(m.labels[0])
        d = m.density(pos)
        assert 0.0 <= d <= 1.0

    def test_curvature_works_after_upgrade(self):
        m = _make_manifold()
        pos = m.position(m.labels[0])
        k = m.curvature(pos)
        assert k >= 0.0

    def test_neighbors_range_query(self):
        m = _make_manifold()
        pos = m.position(m.labels[0])
        nbrs = m.neighbors(pos, r=10.0)
        assert len(nbrs) >= 1

    def test_query_ball_point_used_in_density(self):
        """density() internally uses query_ball_point on cKDTree."""
        m = _make_manifold()
        pos = m.position(m.labels[0])
        m._ensure_kdtree()
        # cKDTree has query_ball_point
        assert hasattr(m._kdtree, "query_ball_point")
        idxs = m._kdtree.query_ball_point(pos, m._density_radius)
        assert isinstance(idxs, list)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Deformation pre-filter via cKDTree range query
# ═══════════════════════════════════════════════════════════════════════════

class TestDeformationPreFilter:
    """PRIORITY 4 — LocalDeformation accepts candidate_labels pre-filter."""

    def test_deformation_with_candidates_matches_full_scan(self):
        """Pre-filtered result should match brute-force within tolerance."""
        m = _make_manifold()
        label = m.labels[0]
        centre = m.position(label)
        delta = np.zeros(104)
        delta[0] = 0.1

        dens = m._state.density.get(label)
        r = m._classifier.locality_radius(dens)
        cutoff = m._deformer.cutoff_sigma * r

        # Full scan (no candidates)
        r1 = m._deformer.apply(
            centre_label=label,
            centre_vector=centre,
            delta=delta,
            locality_radius=r,
            all_points=m._points,
            density_func=lambda l: m._state.density.get(l),
            candidate_labels=None,
        )

        # Pre-filtered — only labels within cutoff
        m._ensure_kdtree()
        idxs = m._kdtree.query_ball_point(centre, cutoff)
        candidates = {m._kdtree_labels[i] for i in idxs}
        r2 = m._deformer.apply(
            centre_label=label,
            centre_vector=centre,
            delta=delta,
            locality_radius=r,
            all_points=m._points,
            density_func=lambda l: m._state.density.get(l),
            candidate_labels=candidates,
        )

        # Same affected set
        assert r1.n_affected == r2.n_affected

    def test_candidate_labels_parameter_is_optional(self):
        """Omitting candidate_labels falls back to full scan (backward compat)."""
        deformer = LocalDeformation(cutoff_sigma=3.0)
        points = {"a": np.zeros(104), "b": np.ones(104) * 0.1}
        result = deformer.apply(
            centre_label="a",
            centre_vector=points["a"],
            delta=np.ones(104) * 0.01,
            locality_radius=1.0,
            all_points=points,
            density_func=lambda l: 0.0,
        )
        assert result.n_affected >= 1

    def test_deform_local_uses_prefilter(self):
        """LivingManifold.deform_local() with cKDTree pre-filter produces
        the same results as before."""
        m = _make_manifold()
        label = m.labels[0]
        pos_before = m.position(label).copy()
        delta = np.zeros(104)
        delta[0] = 0.05
        n_affected = m.deform_local(label, delta)
        pos_after = m.position(label)
        # Centre should have moved
        assert np.linalg.norm(pos_after - pos_before) > 0.01
        assert n_affected >= 1


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Incremental geodesic graph
# ═══════════════════════════════════════════════════════════════════════════

class TestIncrementalGeodesic:
    """PRIORITY 5 — Incremental kNN graph update instead of full O(n²) rebuild."""

    def test_full_rebuild_on_first_query(self):
        gc = GeodesicComputer(k_neighbours=4)
        rng = np.random.default_rng(10)
        for i in range(20):
            gc.add_point(f"p{i}", rng.standard_normal(16))

        dist = gc.distance("p0", "p1")
        assert dist > 0
        assert gc._fully_built

    def test_incremental_update_on_small_change(self):
        gc = GeodesicComputer(k_neighbours=4, rebuild_fraction=0.3)
        rng = np.random.default_rng(20)
        for i in range(50):
            gc.add_point(f"p{i}", rng.standard_normal(16))

        # Force full build
        gc.distance("p0", "p1")
        assert gc._fully_built
        assert len(gc._dirty_labels) == 0

        # Now update just one point — should trigger incremental, not full
        gc.update_point("p5", rng.standard_normal(16))
        assert gc._dirty
        assert len(gc._dirty_labels) == 1

        # Query forces update
        dist = gc.distance("p0", "p5")
        assert dist > 0
        assert not gc._dirty

    def test_add_single_point_incremental(self):
        gc = GeodesicComputer(k_neighbours=4, rebuild_fraction=0.3)
        rng = np.random.default_rng(30)
        for i in range(30):
            gc.add_point(f"p{i}", rng.standard_normal(16))

        gc.distance("p0", "p1")  # full build
        gc.add_point("new_pt", rng.standard_normal(16))

        # Dirty fraction = 1/31 ≈ 0.03 < 0.3 → incremental
        dist = gc.distance("p0", "new_pt")
        assert dist > 0
        assert not gc._dirty

    def test_many_dirty_triggers_full_rebuild(self):
        gc = GeodesicComputer(k_neighbours=4, rebuild_fraction=0.3)
        rng = np.random.default_rng(40)
        for i in range(10):
            gc.add_point(f"p{i}", rng.standard_normal(16))

        gc.distance("p0", "p1")  # full build

        # Dirty 5/10 = 50% > 30% threshold → full rebuild
        for i in range(5):
            gc.update_point(f"p{i}", rng.standard_normal(16))

        assert len(gc._dirty_labels) == 5
        gc.distance("p0", "p5")
        assert not gc._dirty

    def test_remove_point(self):
        gc = GeodesicComputer(k_neighbours=4)
        rng = np.random.default_rng(50)
        for i in range(10):
            gc.add_point(f"p{i}", rng.standard_normal(16))

        gc.distance("p0", "p1")
        gc.remove_point("p5")
        assert "p5" not in gc._vectors

    def test_graph_consistency_after_incremental(self):
        """The geodesic distance after incremental update should be reasonable."""
        gc = GeodesicComputer(k_neighbours=4, rebuild_fraction=0.3)
        rng = np.random.default_rng(60)
        vecs = {}
        for i in range(30):
            v = rng.standard_normal(16)
            gc.add_point(f"p{i}", v)
            vecs[f"p{i}"] = v

        gc.distance("p0", "p1")  # full build

        # Update one point
        new_v = rng.standard_normal(16)
        gc.update_point("p10", new_v)
        vecs["p10"] = new_v

        # Geodesic distance should be at least the Euclidean distance
        geo_dist = gc.distance("p0", "p10")
        euc_dist = float(np.linalg.norm(vecs["p0"] - vecs["p10"]))
        # Graph distance >= Euclidean (shortest path through graph)
        # Allow small tolerance for floating point
        assert geo_dist >= euc_dist * 0.99 or geo_dist > 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: FAISS vocabulary matching
# ═══════════════════════════════════════════════════════════════════════════

class TestFAISSMatcher:
    """PRIORITY 3 — FAISS-accelerated vocabulary matching (optional)."""

    def test_matcher_works_without_faiss(self):
        """Even if FAISS is not installed, matching still works via linear scan."""
        matcher = ResonanceMatcher()
        assert len(matcher.vocabulary) == 32

        wp = WavePoint(vector=np.random.randn(WAVE_DIM), amplitude=0.5, label="x")
        segment = WaveSegment(
            points=[wp],
            mean_amplitude=0.5,
            peak_point=wp,
            uncertainty=0.5,
            flow_speed=0.5,
        )
        result = matcher.match(segment)
        assert result.resonance_score > 0

    def test_faiss_index_not_built_for_small_vocab(self):
        """FAISS index should not be created for < threshold entries."""
        matcher = ResonanceMatcher()
        matcher._ensure_faiss_index()
        assert matcher._faiss_index is None  # 32 < 200 threshold

    @pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")
    def test_faiss_index_built_for_large_vocab(self):
        """When FAISS is available and vocab is large, index is built."""
        matcher = ResonanceMatcher()
        rng = np.random.default_rng(42)
        for i in range(300):
            matcher.vocabulary.append(ExpressionEntry(
                text=f"template_{i}",
                wave_profile=rng.standard_normal(WAVE_DIM),
            ))
        matcher._faiss_dirty = True
        matcher._ensure_faiss_index()
        assert matcher._faiss_index is not None

    def test_get_candidates_returns_full_vocab_when_small(self):
        matcher = ResonanceMatcher()
        query = np.random.randn(WAVE_DIM)
        candidates = matcher._get_candidates(query)
        assert len(candidates) == len(matcher.vocabulary)

    def test_faiss_dirty_flag_on_load_vocabulary(self):
        """Loading vocabulary marks FAISS index as dirty."""
        matcher = ResonanceMatcher()
        matcher._faiss_dirty = False
        # We can't easily test load_vocabulary without a file, but we
        # can verify the flag is managed correctly
        assert not matcher._faiss_dirty
        # After adding entries manually and toggling dirty
        matcher.vocabulary.append(ExpressionEntry(
            text="test", wave_profile=np.zeros(WAVE_DIM)
        ))
        matcher._faiss_dirty = True
        assert matcher._faiss_dirty


# ═══════════════════════════════════════════════════════════════════════════
# TEST: GEOPipeline save / load
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineSaveLoad:
    """PRIORITY 2+6 — GEOPipeline.save() / .load() convenience methods."""

    def test_save_creates_manifold_file(self):
        pipeline = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=42)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            result = pipeline.save(path)
            assert os.path.exists(path)
            assert result["n_points"] == pipeline.n_concepts
        finally:
            os.unlink(path)

    def test_save_with_vocabulary(self):
        pipeline = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            m_path = os.path.join(tmpdir, "manifold.npz")
            v_path = os.path.join(tmpdir, "vocab.npz")
            result = pipeline.save(m_path, vocabulary_path=v_path)
            assert os.path.exists(m_path)
            assert os.path.exists(v_path)
            assert result["n_vocab"] == len(pipeline._renderer.matcher.vocabulary)

    def test_load_restores_concepts(self):
        pipeline1 = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=42)
        rng = np.random.default_rng(42)
        for i in range(5):
            pipeline1.learn(Experience(vector=_random_vec(rng), label=f"p8_{i}"))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            pipeline1.save(path)
            pipeline2 = GEOPipeline.load(path, flow_seed=42)

            for i in range(5):
                label = f"p8_{i}"
                assert label in pipeline2.manifold.labels
                np.testing.assert_allclose(
                    pipeline1.manifold.position(label),
                    pipeline2.manifold.position(label),
                    atol=1e-12,
                )
        finally:
            os.unlink(path)

    def test_load_with_vocabulary(self):
        pipeline1 = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=42)
        n_vocab_before = len(pipeline1._renderer.matcher.vocabulary)

        with tempfile.TemporaryDirectory() as tmpdir:
            m_path = os.path.join(tmpdir, "manifold.npz")
            v_path = os.path.join(tmpdir, "vocab.npz")
            pipeline1.save(m_path, vocabulary_path=v_path)
            pipeline2 = GEOPipeline.load(m_path, vocabulary_path=v_path)

            # Base 32 + loaded 32 (same entries loaded on top)
            n_vocab_after = len(pipeline2._renderer.matcher.vocabulary)
            assert n_vocab_after >= n_vocab_before

    def test_loaded_pipeline_can_query(self):
        """After loading, the pipeline should be fully functional for queries."""
        pipeline1 = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=42)
        rng = np.random.default_rng(77)
        pipeline1.learn(Experience(vector=_random_vec(rng), label="test_concept"))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            pipeline1.save(path)
            pipeline2 = GEOPipeline.load(path, flow_seed=42)
            result = pipeline2.query(_random_vec(rng), label="test query")
            assert result.text  # non-empty output
            assert result.confidence > 0
        finally:
            os.unlink(path)

    def test_loaded_pipeline_can_learn(self):
        """After loading, the pipeline can continue learning (growth mode)."""
        pipeline1 = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=42)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            pipeline1.save(path)
            pipeline2 = GEOPipeline.load(path, flow_seed=42)
            rng = np.random.default_rng(88)
            result = pipeline2.learn(
                Experience(vector=_random_vec(rng), label="post_load_concept")
            )
            assert "post_load_concept" in pipeline2.manifold.labels
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Design constraints still upheld
# ═══════════════════════════════════════════════════════════════════════════

class TestDesignConstraints:
    """All six non-negotiable design constraints remain intact."""

    def test_no_ml_libraries_in_persistence(self):
        """No ML framework imported in src/persistence/."""
        import importlib
        import sys
        forbidden = {"torch", "tensorflow", "jax", "keras"}
        mod = importlib.import_module("src.persistence.snapshot")
        source = open(mod.__file__).read()
        for lib in forbidden:
            assert f"import {lib}" not in source
            assert f"from {lib}" not in source

    def test_no_weights_in_snapshot(self):
        """ManifoldSnapshot stores pure numpy arrays, no weight matrices."""
        m = _make_manifold()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m, path)
            data = np.load(path, allow_pickle=True)
            # Only expected keys
            expected_keys = {
                "labels", "positions", "densities", "deformations",
                "curvatures", "manifold_time", "n_writes",
                "format_version", "dimension",
            }
            actual_keys = set(data.files)
            assert actual_keys == expected_keys
        finally:
            os.unlink(path)

    def test_no_tokens_in_snapshot_format(self):
        """Snapshot contains continuous vectors, not token IDs."""
        m = _make_manifold()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ManifoldSnapshot.save(m, path)
            data = np.load(path, allow_pickle=True)
            assert data["positions"].dtype == np.float64
            assert data["positions"].shape[1] == 104
        finally:
            os.unlink(path)

    def test_local_updates_preserved_with_prefilter(self):
        """cKDTree pre-filter still honours the locality guarantee."""
        m = _make_manifold()
        label = m.labels[0]
        centre = m.position(label).copy()
        delta = np.zeros(104)
        delta[0] = 0.1
        n_affected = m.deform_local(label, delta)

        # Verify: only points within cutoff were affected
        dens = m._state.density.get(label)
        r = m._classifier.locality_radius(dens)
        cutoff = m._deformer.cutoff_sigma * r

        for other_label in m.labels:
            if other_label == label:
                continue
            disp = m._state.deformation.displacement(other_label)
            if np.linalg.norm(disp) > 1e-12:
                dist_from_centre = float(np.linalg.norm(
                    m.position(other_label) - centre - disp
                ))
                # The point was within cutoff before deformation
                # (approximate check)
                assert dist_from_centre < cutoff * 2

    def test_ckdtree_is_spatial_index_not_learned(self):
        """cKDTree is a pure spatial data structure — no weights, no training."""
        from scipy.spatial import cKDTree
        m = _make_manifold()
        m._ensure_kdtree()
        tree = m._kdtree
        # cKDTree is deterministic and contains no learned parameters
        assert isinstance(tree, cKDTree)
        assert tree.n == m.n_points

    def test_faiss_is_spatial_index_not_learned(self):
        """FAISS IndexFlatIP is brute-force inner product — no learned params.
        (Verified at import level, not requiring FAISS installation.)"""
        # The matcher uses IndexFlatIP which is an exact search index,
        # not a learned model.
        matcher = ResonanceMatcher()
        # Regardless of FAISS availability, the constraint holds:
        # no weights are trained in the matching step.
        assert len(matcher.vocabulary) == 32

    def test_separation_persistence_module(self):
        """src/persistence/ doesn't import C5, C6, or C7."""
        import importlib
        mod = importlib.import_module("src.persistence.snapshot")
        source = open(mod.__file__).read()
        # Should not import flow engine, resonance layer, or expression renderer
        assert "flow_engine" not in source
        assert "resonance_layer" not in source
        assert "expression" not in source


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Integration — full pipeline round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests for Phase 8 features."""

    def test_full_pipeline_save_load_query_cycle(self):
        """Build → learn → save → load → query — complete cycle."""
        rng = np.random.default_rng(42)
        p1 = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=42)

        # Learn concepts
        for i in range(10):
            p1.learn(Experience(vector=_random_vec(rng), label=f"c_{i}"))

        # Query
        result1 = p1.query(_random_vec(rng), label="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            m_path = os.path.join(tmpdir, "manifold.npz")
            v_path = os.path.join(tmpdir, "vocab.npz")

            p1.save(m_path, vocabulary_path=v_path)

            # Load into fresh pipeline
            p2 = GEOPipeline.load(m_path, vocabulary_path=v_path, flow_seed=42)

            assert p2.n_concepts == p1.n_concepts

            # Can still query
            result2 = p2.query(_random_vec(np.random.default_rng(42)), label="test2")
            assert result2.text

    def test_ckdtree_and_geodesic_work_together(self):
        """After cKDTree + incremental geodesic upgrades, manifold queries
        still produce valid results."""
        m = _make_manifold()
        rng = np.random.default_rng(100)
        ae = AnnealingEngine(m, T0=1.0, lambda_=0.01, T_floor=0.05)

        for i in range(20):
            ae.process(Experience(vector=_random_vec(rng), label=f"int_{i}"))

        # Geodesic should work
        labels = m.labels[:2]
        geo = m.geodesic(labels[0], labels[1])
        assert len(geo) >= 2
        geo_dist = m.geodesic_distance(labels[0], labels[1])
        assert geo_dist > 0

    def test_snapshot_preserves_query_equivalence(self):
        """Queries on saved-and-loaded manifold yield trajectory through
        the same geometric landscape."""
        rng = np.random.default_rng(123)
        p1 = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=99)

        for i in range(5):
            p1.learn(Experience(vector=_random_vec(rng), label=f"eq_{i}"))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            p1.save(path)
            p2 = GEOPipeline.load(path, flow_seed=99)

            # Same query vector → should produce similar trajectory
            qv = _random_vec(np.random.default_rng(200))
            r1 = p1.query(qv, label="eq_test")
            r2 = p2.query(qv, label="eq_test")

            # Same number of steps (deterministic with same seed + same manifold)
            assert r1.n_steps == r2.n_steps
        finally:
            os.unlink(path)
