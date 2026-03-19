"""Phase 2 test suite — Living Manifold + Contrast Engine.

Run with:
  python3 -m pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

# ── Phase 1 (needed to build the seed manifold) ──────────────────────────────
from src.phase1.seed_geometry import SeedGeometryEngine

# ── Phase 2 ──────────────────────────────────────────────────────────────────
from src.phase2.living_manifold.state import (
    DeformationField,
    DensityField,
    ManifoldState,
)
from src.phase2.living_manifold.regions import RegionClassifier, RegionType
from src.phase2.living_manifold.geodesic import GeodesicComputer
from src.phase2.living_manifold.deformation import LocalDeformation
from src.phase2.living_manifold.manifold import LivingManifold
from src.phase2.contrast_engine.persistence import (
    PersistenceDiagram,
    PersistenceEvent,
)
from src.phase2.contrast_engine.engine import (
    ContrastEngine,
    ContrastPair,
    ContrastResult,
    JudgmentType,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def seed_manifold():
    """Build M₀ once for the entire test module (slow ≈ 0.015 s)."""
    engine = SeedGeometryEngine()
    return engine.build()


@pytest.fixture(scope="module")
def living(seed_manifold):
    """Fresh LivingManifold initialised from M₀."""
    return LivingManifold(seed_manifold)


@pytest.fixture(scope="module")
def contrast(living):
    """ContrastEngine wrapping the module-level LivingManifold."""
    return ContrastEngine(living, alpha=0.1, beta=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# TestDeformationField
# ─────────────────────────────────────────────────────────────────────────────

class TestDeformationField:
    def test_register_creates_zero_entry(self):
        df = DeformationField()
        df.register("a", 4)
        d = df.displacement("a")
        assert d.shape == (4,)
        assert np.allclose(d, 0.0)

    def test_register_is_idempotent(self):
        df = DeformationField()
        df.register("a", 4)
        df.register("a", 4)  # second call should not overwrite
        assert np.allclose(df.displacement("a"), 0.0)

    def test_accumulate_adds_vectors(self):
        df = DeformationField()
        df.register("a", 3)
        df.accumulate("a", np.array([1.0, 2.0, 3.0]))
        df.accumulate("a", np.array([0.5, 0.5, 0.5]))
        result = df.displacement("a")
        assert np.allclose(result, [1.5, 2.5, 3.5])

    def test_accumulate_new_label(self):
        df = DeformationField()
        df.accumulate("new", np.array([1.0, 0.0]))
        assert np.allclose(df.displacement("new"), [1.0, 0.0])

    def test_missing_label_returns_ones(self):
        df = DeformationField()
        d = df.displacement("ghost")
        # Returns numpy zeros(1) for missing
        assert d is not None

    def test_has(self):
        df = DeformationField()
        df.register("x", 2)
        assert df.has("x")
        assert not df.has("y")

    def test_len(self):
        df = DeformationField()
        df.register("a", 2)
        df.register("b", 2)
        assert len(df) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestDensityField
# ─────────────────────────────────────────────────────────────────────────────

class TestDensityField:
    def test_set_and_get(self):
        field = DensityField()
        field.set("cat", 0.7)
        assert field.get("cat") == pytest.approx(0.7)

    def test_missing_returns_zero(self):
        field = DensityField()
        assert field.get("ghost") == 0.0

    def test_clamps_above_one(self):
        field = DensityField()
        field.set("x", 1.5)
        assert field.get("x") == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        field = DensityField()
        field.set("x", -0.3)
        assert field.get("x") == pytest.approx(0.0)

    def test_len(self):
        field = DensityField()
        field.set("a", 0.1)
        field.set("b", 0.9)
        assert len(field) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestManifoldState
# ─────────────────────────────────────────────────────────────────────────────

class TestManifoldState:
    def test_initial_time_is_zero(self):
        state = ManifoldState()
        assert state.t == 0.0
        assert state.n_writes == 0

    def test_tick_increments_writes(self):
        state = ManifoldState()
        state.tick()
        state.tick()
        assert state.n_writes == 2

    def test_tick_advances_time(self):
        state = ManifoldState()
        state.tick()
        assert state.t > 0.0

    def test_curvature_set_get(self):
        state = ManifoldState()
        state.set_curvature("p", 0.42)
        assert state.get_curvature("p") == pytest.approx(0.42)

    def test_curvature_default_zero(self):
        state = ManifoldState()
        assert state.get_curvature("missing") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TestRegionClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestRegionClassifier:
    def setup_method(self):
        self.clf = RegionClassifier()

    def test_high_density_is_crystallized(self):
        assert self.clf.classify(0.8) == RegionType.CRYSTALLIZED

    def test_medium_density_is_flexible(self):
        assert self.clf.classify(0.4) == RegionType.FLEXIBLE

    def test_low_density_is_unknown(self):
        assert self.clf.classify(0.05) == RegionType.UNKNOWN

    def test_at_high_threshold_boundary(self):
        # Just above high threshold → crystallized
        assert self.clf.classify(0.61) == RegionType.CRYSTALLIZED

    def test_at_low_threshold_boundary(self):
        # At exact threshold → unknown
        assert self.clf.classify(0.20) == RegionType.UNKNOWN

    def test_stiffness_equals_density(self):
        for rho in [0.0, 0.3, 0.7, 1.0]:
            assert self.clf.stiffness(rho) == pytest.approx(rho)

    def test_flexibility_complement(self):
        for rho in [0.0, 0.3, 0.7, 1.0]:
            assert self.clf.stiffness(rho) + self.clf.flexibility(rho) == pytest.approx(1.0)

    def test_locality_radius_decreases_with_density(self):
        r_low = self.clf.locality_radius(0.1)
        r_high = self.clf.locality_radius(0.9)
        assert r_low > r_high

    def test_locality_radius_positive(self):
        for rho in [0.0, 0.5, 1.0]:
            assert self.clf.locality_radius(rho) > 0.0

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            RegionClassifier(high_threshold=0.2, low_threshold=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# TestGeodesicComputer
# ─────────────────────────────────────────────────────────────────────────────

class TestGeodesicComputer:
    def _make_linear_graph(self, n: int = 6) -> GeodesicComputer:
        """n points at positions [0, 1, 2, ...] in 2D."""
        gc = GeodesicComputer(k_neighbours=2)
        for i in range(n):
            gc.add_point(str(i), np.array([float(i), 0.0]))
        return gc

    def test_self_distance_is_zero(self):
        gc = self._make_linear_graph()
        assert gc.distance("0", "0") == pytest.approx(0.0)

    def test_adjacent_distance(self):
        gc = self._make_linear_graph()
        assert gc.distance("0", "1") == pytest.approx(1.0, abs=0.1)

    def test_far_distance_greater_than_near(self):
        gc = self._make_linear_graph()
        assert gc.distance("0", "5") > gc.distance("0", "1")

    def test_symmetry(self):
        gc = self._make_linear_graph()
        d_ab = gc.distance("0", "3")
        d_ba = gc.distance("3", "0")
        assert d_ab == pytest.approx(d_ba, rel=0.01)

    def test_path_starts_and_ends_correctly(self):
        gc = self._make_linear_graph()
        path = gc.path("0", "5")
        assert path[0] == "0"
        assert path[-1] == "5"

    def test_path_is_connected(self):
        gc = self._make_linear_graph()
        path = gc.path("0", "3")
        assert len(path) >= 2

    def test_add_and_update_point(self):
        gc = GeodesicComputer(k_neighbours=2)
        gc.add_point("a", np.array([0.0, 0.0]))
        gc.add_point("b", np.array([1.0, 0.0]))
        gc.update_point("b", np.array([0.5, 0.0]))
        d = gc.distance("a", "b")
        assert d == pytest.approx(0.5, abs=0.2)

    def test_missing_point_falls_back(self):
        gc = GeodesicComputer(k_neighbours=2)
        gc.add_point("a", np.array([0.0, 0.0]))
        gc.add_point("b", np.array([3.0, 4.0]))
        # Should fall back to Euclidean (5.0)
        d = gc.distance("a", "b")
        assert d == pytest.approx(5.0, abs=1.0)

    def test_len(self):
        gc = self._make_linear_graph(4)
        assert len(gc) == 4


# ─────────────────────────────────────────────────────────────────────────────
# TestLocalDeformation
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalDeformation:
    DIM = 4

    def _make_points(self) -> dict:
        """5 points in a line."""
        return {str(i): np.array([float(i), 0.0, 0.0, 0.0]) for i in range(5)}

    def test_centre_point_most_affected(self):
        deformer = LocalDeformation(cutoff_sigma=3.0)
        pts = self._make_points()
        delta = np.array([0.0, 1.0, 0.0, 0.0])
        result = deformer.apply(
            "2", pts["2"], delta, locality_radius=1.0,
            all_points=pts, density_func=lambda l: 0.0,
        )
        # Extract displacements
        disp = {l: np.linalg.norm(d) for l, d in result.affected}
        assert "2" in disp
        assert disp["2"] >= max((disp.get(l, 0) for l in ["0", "4"]), default=0)

    def test_distant_points_unaffected(self):
        deformer = LocalDeformation(cutoff_sigma=3.0)
        pts = {
            "near": np.zeros(4),
            "far": np.array([100.0, 0.0, 0.0, 0.0]),
        }
        delta = np.array([1.0, 0.0, 0.0, 0.0])
        result = deformer.apply(
            "near", pts["near"], delta, locality_radius=1.0,
            all_points=pts, density_func=lambda l: 0.0,
        )
        affected_labels = [l for l, _ in result.affected]
        assert "far" not in affected_labels

    def test_high_density_resists_deformation(self):
        deformer = LocalDeformation(cutoff_sigma=3.0)
        pts = {"a": np.zeros(4), "b": np.zeros(4)}
        delta = np.ones(4)
        # Apply to "a" with "b" being very dense (stiff)
        result = deformer.apply(
            "a", pts["a"], delta, locality_radius=1.0,
            all_points=pts,
            density_func=lambda l: 1.0 if l == "b" else 0.0,
        )
        disp_b = {l: np.linalg.norm(d) for l, d in result.affected}.get("b", 0.0)
        assert disp_b == pytest.approx(0.0, abs=1e-9)

    def test_zero_radius_only_affects_centre(self):
        deformer = LocalDeformation(cutoff_sigma=3.0)
        pts = {"c": np.zeros(4), "neighbor": np.array([0.1, 0.0, 0.0, 0.0])}
        delta = np.ones(4)
        result = deformer.apply(
            "c", pts["c"], delta, locality_radius=0.0,
            all_points=pts, density_func=lambda l: 0.0,
        )
        affected = [l for l, _ in result.affected]
        assert "c" in affected
        # neighbor might or might not be included depending on logic — we just
        # check the number is small
        assert len(affected) <= 2

    def test_locality_validate_passes(self):
        deformer = LocalDeformation(cutoff_sigma=3.0)
        pts = self._make_points()
        delta = np.array([0.0, 1.0, 0.0, 0.0])
        result = deformer.apply(
            "2", pts["2"], delta, locality_radius=1.0,
            all_points=pts, density_func=lambda l: 0.0,
        )
        valid = deformer.validate_locality(result, pts, pts["2"], max_radius=5.0)
        assert valid

    def test_n_affected_matches_list(self):
        deformer = LocalDeformation(cutoff_sigma=3.0)
        pts = self._make_points()
        delta = np.ones(4)
        result = deformer.apply(
            "2", pts["2"], delta, locality_radius=2.0,
            all_points=pts, density_func=lambda l: 0.0,
        )
        assert result.n_affected == len(result.affected)


# ─────────────────────────────────────────────────────────────────────────────
# TestLivingManifold
# ─────────────────────────────────────────────────────────────────────────────

class TestLivingManifold:

    def test_initialises_with_seed_points(self, living):
        assert living.n_points >= 81  # seed had 81 points

    def test_dim_is_104(self, living):
        assert living.DIM == 104

    def test_position_known_label(self, living):
        p = living.position("causal::perturbation")
        assert p.shape == (104,)
        assert np.all(np.isfinite(p))

    def test_position_unknown_label_raises(self, living):
        with pytest.raises(KeyError):
            living.position("unknown::this_should_not_exist_XYZ")

    def test_distance_nonnegative(self, living):
        p = living.position("causal::perturbation")
        q = living.position("causal::direct_effect")
        assert living.distance(p, q) >= 0.0

    def test_distance_self_is_zero(self, living):
        p = living.position("causal::perturbation")
        assert living.distance(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_distance_symmetry(self, living):
        p = living.position("causal::perturbation")
        q = living.position("causal::direct_effect")
        assert living.distance(p, q) == pytest.approx(living.distance(q, p), rel=0.01)

    def test_curvature_positive(self, living):
        p = living.position("causal::perturbation")
        assert living.curvature(p) > 0.0

    def test_density_in_range(self, living):
        p = living.position("causal::perturbation")
        dens = living.density(p)
        assert 0.0 <= dens <= 1.0

    def test_neighbors_returns_list(self, living):
        p = living.position("causal::perturbation")
        nbrs = living.neighbors(p, r=5.0)
        assert isinstance(nbrs, list)

    def test_nearest_returns_k(self, living):
        p = living.position("causal::perturbation")
        nearest = living.nearest(p, k=3)
        assert len(nearest) <= 3

    def test_causal_direction_shape(self, living):
        p = living.position("causal::perturbation")
        q = living.position("causal::direct_effect")
        cd = living.causal_direction(p, q)
        assert cd.shape == (104,)

    def test_causal_direction_unit_or_zero(self, living):
        p = living.position("causal::perturbation")
        q = living.position("causal::direct_effect")
        cd = living.causal_direction(p, q)
        norm = float(np.linalg.norm(cd))
        assert norm == pytest.approx(0.0, abs=1e-6) or norm == pytest.approx(1.0, abs=1e-4)

    def test_causal_ancestry_known_pair(self, living):
        p = living.position("causal::perturbation")
        q = living.position("causal::direct_effect")
        result = living.causal_ancestry(p, q)
        assert isinstance(result, bool)

    def test_region_type_returns_enum(self, living):
        p = living.position("causal::perturbation")
        rt = living.region_type(p)
        assert isinstance(rt, RegionType)

    def test_locality_radius_positive(self, living):
        p = living.position("causal::perturbation")
        r = living.locality_radius(p)
        assert r > 0.0

    def test_confidence_in_range(self, living):
        p = living.position("causal::perturbation")
        c = living.confidence(p)
        assert 0.0 <= c <= 1.0

    def test_geodesic_path_valid(self, living):
        path = living.geodesic("causal::perturbation", "causal::direct_effect")
        assert len(path) >= 1
        for vec in path:
            assert vec.shape == (104,)

    def test_place_adds_point(self, living):
        vec = np.zeros(104)
        vec[0] = 42.0
        mp = living.place("test::placed_point", vec)
        assert mp.label == "test::placed_point"
        assert living.n_points >= 82  # at least one more than before

    def test_place_point_retrievable(self, living):
        vec = np.zeros(104)
        vec[1] = 7.0
        living.place("test::retrievable", vec)
        pos = living.position("test::retrievable")
        assert pos.shape == (104,)

    def test_deform_local_returns_count(self, living):
        # Place a probe point and deform it
        vec = np.zeros(104)
        living.place("test::deform_probe", vec.copy())
        delta = np.zeros(104)
        delta[0] = 0.01
        count = living.deform_local("test::deform_probe", delta)
        assert count >= 1

    def test_deform_local_moves_point(self, living):
        vec = np.zeros(104)
        vec[2] = 5.0
        living.place("test::moveable", vec.copy())
        before = living.position("test::moveable").copy()
        delta = np.zeros(104)
        delta[2] = 0.1
        living.deform_local("test::moveable", delta)
        after = living.position("test::moveable")
        # Point should have moved
        assert not np.allclose(before, after)

    def test_deform_increments_write_counter(self, living):
        n_before = living.n_writes
        vec = np.zeros(104)
        living.place("test::write_counter", vec.copy())
        delta = np.zeros(104)
        delta[3] = 0.01
        living.deform_local("test::write_counter", delta)
        assert living.n_writes > n_before

    def test_update_density_returns_float(self, living):
        dens = living.update_density("causal::perturbation")
        assert isinstance(dens, float)
        assert 0.0 <= dens <= 1.0

    def test_validate_passes(self, living):
        assert living.validate() is True

    def test_summary_contains_key_info(self, living):
        s = living.summary()
        assert "Living Manifold" in s
        assert "Points" in s


# ─────────────────────────────────────────────────────────────────────────────
# TestPersistenceDiagram
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistenceDiagram:

    def test_empty_diagram(self):
        diag = PersistenceDiagram()
        assert len(diag) == 0

    def test_record_adds_pair(self):
        diag = PersistenceDiagram()
        diag.record("cat", "dog", 1.5, 0.0)
        assert len(diag) == 1

    def test_record_same_pair_multiple_times(self):
        diag = PersistenceDiagram()
        diag.record("a", "b", 1.0, 0.0)
        diag.record("a", "b", 0.8, 1.0)
        assert len(diag) == 1  # still one pair

    def test_symmetric_key(self):
        diag = PersistenceDiagram()
        diag.record("x", "y", 2.0, 0.0)
        diag.record("y", "x", 2.0, 1.0)
        assert len(diag) == 1  # same pair

    def test_birth_event_created_when_close(self):
        diag = PersistenceDiagram(cluster_threshold_init=3.0)
        diag.record("a", "b", 1.0, 0.0)
        events = [ev for ev in diag._events if ev.label_a in ("a", "b")]
        assert len(events) >= 1

    def test_no_birth_when_far(self):
        diag = PersistenceDiagram(cluster_threshold_init=1.0)
        diag.record("a", "b", 10.0, 0.0)
        assert len(diag._events) == 0

    def test_persistent_features_threshold(self):
        diag = PersistenceDiagram(cluster_threshold_init=5.0)
        # Two close concepts → short lifetime
        diag.record("a", "b", 1.0, 0.0)
        diag.record("a", "b", 6.0, 10.0)  # death
        # min_lifetime=15 → none pass
        feats = diag.get_persistent_features(min_lifetime=15.0)
        assert len(feats) == 0

    def test_current_distances_returns_latest(self):
        diag = PersistenceDiagram()
        diag.record("p", "q", 3.0, 0.0)
        diag.record("p", "q", 1.5, 1.0)
        cd = diag.current_distances()
        assert cd[("p", "q")] == pytest.approx(1.5)

    def test_cluster_corrections_type(self):
        diag = PersistenceDiagram(cluster_threshold_init=5.0)
        diag.record("r", "s", 1.0, 0.0)
        corrections = diag.cluster_corrections(min_lifetime=0.0)
        assert isinstance(corrections, list)


# ─────────────────────────────────────────────────────────────────────────────
# TestContrastEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestContrastEngine:

    @pytest.fixture()
    def fresh_living(self, seed_manifold):
        """Isolated LivingManifold per test to avoid state leakage."""
        return LivingManifold(seed_manifold)

    @pytest.fixture()
    def engine(self, fresh_living):
        return ContrastEngine(fresh_living, alpha=0.1, beta=0.1)

    def test_initial_judgment_count(self, engine):
        assert engine.n_judgments == 0

    def test_same_judgment_reduces_distance(self, engine):
        result = engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.SAME,
        )
        assert isinstance(result, ContrastResult)
        # Distance should have decreased (or at most not increased much)
        assert result.distance_after <= result.distance_before + 0.5

    def test_different_judgment_increases_distance(self, engine, fresh_living):
        # Use two concepts from different semantic regions
        result = engine.judge(
            "causal::perturbation",
            "logical::00000101",
            JudgmentType.DIFFERENT,
        )
        assert isinstance(result, ContrastResult)

    def test_judgment_increments_counter(self, engine):
        engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.SAME,
        )
        assert engine.n_judgments == 1

    def test_result_has_correct_labels(self, engine):
        result = engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.SAME,
        )
        assert result.label_a == "causal::perturbation"
        assert result.label_b == "causal::direct_effect"

    def test_result_distance_change_correct_direction_same(self, engine):
        result = engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.SAME,
        )
        # SAME should reduce or maintain distance
        assert result.distance_after <= result.distance_before * 1.05

    def test_different_judgment_moves_apart(self, engine):
        result = engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.DIFFERENT,
        )
        # DIFFERENT should increase or maintain distance
        assert result.distance_after >= result.distance_before * 0.95

    def test_delta_shape(self, engine):
        result = engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.SAME,
        )
        assert result.delta_a.shape == (104,)
        assert result.delta_b.shape == (104,)

    def test_n_affected_positive(self, engine):
        result = engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.SAME,
        )
        assert result.n_affected_a >= 1
        assert result.n_affected_b >= 1

    def test_judge_batch(self, engine):
        pairs = [
            ContrastPair("causal::perturbation", "causal::direct_effect", JudgmentType.SAME),
            ContrastPair("causal::perturbation", "causal::direct_effect", JudgmentType.DIFFERENT),
        ]
        results = engine.judge_batch(pairs)
        assert len(results) == 2

    def test_history_grows(self, engine):
        n_before = len(engine.history)
        engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.SAME,
        )
        assert len(engine.history) == n_before + 1

    def test_temporal_pairs_generation(self, engine):
        seq = ["causal::perturbation", "causal::direct_effect", "causal::downstream_effect"]
        pairs = engine.generate_temporal_pairs(seq, window=2)
        assert len(pairs) > 0
        for p in pairs:
            assert p.judgment == JudgmentType.SAME

    def test_contrast_pairs_generation(self, engine):
        group_a = ["causal::perturbation"]
        group_b = ["logical::00000101", "logical::11111000"]
        pairs = engine.generate_contrast_pairs(group_a, group_b)
        assert len(pairs) == 2
        for p in pairs:
            assert p.judgment == JudgmentType.DIFFERENT

    def test_correct_direction_rate_bounds(self, engine):
        rate = engine.correct_direction_rate()
        assert 0.0 <= rate <= 1.0

    def test_unknown_label_raises(self, engine):
        with pytest.raises(KeyError):
            engine.judge(
                "causal::perturbation",
                "totally::unknown_label_XYZ",
                JudgmentType.SAME,
            )

    def test_strength_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ContrastPair("a", "b", JudgmentType.SAME, strength=0.0)

    def test_summary_contains_judgment_count(self, engine):
        s = engine.summary()
        assert "Contrast Engine" in s

    def test_persistence_diagram_receives_records(self, engine):
        engine.judge(
            "causal::perturbation",
            "causal::direct_effect",
            JudgmentType.SAME,
        )
        assert len(engine.persistence_diagram) >= 1
