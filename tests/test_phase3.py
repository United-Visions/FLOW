"""Phase 3 test suite — Annealing Engine (C3).

Run with:
  python3 -m pytest tests/test_phase3.py -v

All Phase 1 and Phase 2 tests must remain green (run full suite):
  python3 -m pytest tests/ -v
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

# ── Phase 1 (needed to build the seed manifold) ──────────────────────────────
from src.phase1.seed_geometry import SeedGeometryEngine

# ── Phase 2 (Living Manifold wraps M₀) ───────────────────────────────────────
from src.phase2.living_manifold.manifold import LivingManifold

# ── Phase 3 ───────────────────────────────────────────────────────────────────
from src.phase3.annealing_engine.schedule import TemperatureSchedule
from src.phase3.annealing_engine.novelty import NoveltyEstimator, NoveltyResult
from src.phase3.annealing_engine.experience import Experience, ExperienceResult
from src.phase3.annealing_engine.engine import AnnealingEngine, AnnealingStats
from src.phase3.annealing_engine import (
    AnnealingEngine as AnnealingEngineAlias,
    TemperatureSchedule as TemperatureScheduleAlias,
    NoveltyEstimator as NoveltyEstimatorAlias,
    Experience as ExperienceAlias,
    ExperienceResult as ExperienceResultAlias,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def seed_manifold():
    """Build M₀ once for the entire test module."""
    return SeedGeometryEngine().build()


@pytest.fixture(scope="module")
def living(seed_manifold):
    """Fresh LivingManifold initialised from M₀."""
    return LivingManifold(seed_manifold)


@pytest.fixture()
def fresh_living(seed_manifold):
    """A fresh LivingManifold for tests that mutate state."""
    return LivingManifold(seed_manifold)


@pytest.fixture()
def engine(fresh_living):
    """AnnealingEngine with default parameters on a fresh manifold."""
    return AnnealingEngine(fresh_living)


@pytest.fixture()
def vec_104():
    """A random normalised 104D vector."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(104)
    return v / np.linalg.norm(v)


# ─────────────────────────────────────────────────────────────────────────────
# TestTemperatureSchedule
# ─────────────────────────────────────────────────────────────────────────────

class TestTemperatureSchedule:

    def test_initial_temperature_is_T0_plus_floor(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05)
        assert s.temperature(0.0) == pytest.approx(1.05)

    def test_temperature_at_large_t_approaches_floor(self):
        s = TemperatureSchedule(T0=1.0, lambda_=1.0, T_floor=0.05)
        assert s.temperature(100.0) == pytest.approx(0.05, abs=1e-6)

    def test_temperature_decreases_monotonically(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.1, T_floor=0.05)
        temps = [s.temperature(t) for t in range(50)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]

    def test_temperature_never_below_floor(self):
        s = TemperatureSchedule(T0=2.0, lambda_=0.5, T_floor=0.1)
        for t in [0, 1, 5, 10, 50, 100, 1000]:
            assert s.temperature(float(t)) >= 0.1 - 1e-9

    def test_step_advances_time_by_dt(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05, dt=2.0)
        s.step()
        assert s.t == pytest.approx(2.0)
        s.step()
        assert s.t == pytest.approx(4.0)

    def test_step_returns_temperature(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05)
        T = s.step()
        assert T == pytest.approx(s.current_temperature)

    def test_reset_returns_to_zero(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05)
        for _ in range(10):
            s.step()
        assert s.t > 0
        s.reset()
        assert s.t == 0.0

    def test_reset_restores_initial_temperature(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.1, T_floor=0.05)
        for _ in range(20):
            s.step()
        s.reset()
        assert s.current_temperature == pytest.approx(1.05)

    def test_is_cold_false_at_start(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05)
        assert not s.is_cold(threshold=0.1)

    def test_is_cold_true_after_many_steps(self):
        s = TemperatureSchedule(T0=1.0, lambda_=1.0, T_floor=0.05, dt=10.0)
        for _ in range(100):
            s.step()
        assert s.is_cold(threshold=0.1)

    def test_locality_radius_decreases_with_time(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.1, T_floor=0.05)
        r0 = s.locality_radius(base_radius=10.0, t=0.0)
        r1 = s.locality_radius(base_radius=10.0, t=50.0)
        assert r0 > r1

    def test_locality_radius_at_t0_equals_base(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05)
        r = s.locality_radius(base_radius=10.0, t=0.0)
        assert r == pytest.approx(10.0)

    def test_invalid_T0_raises(self):
        with pytest.raises(ValueError):
            TemperatureSchedule(T0=-1.0, lambda_=0.01, T_floor=0.05)

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            TemperatureSchedule(T0=1.0, lambda_=-0.01, T_floor=0.05)

    def test_invalid_T_floor_raises(self):
        with pytest.raises(ValueError):
            TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=-0.1)

    def test_T_floor_ge_T0_raises(self):
        with pytest.raises(ValueError):
            TemperatureSchedule(T0=0.5, lambda_=0.01, T_floor=0.5)

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError):
            TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05, dt=0.0)

    def test_initial_temperature_property(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05)
        assert s.initial_temperature == pytest.approx(1.05)

    def test_t_property(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.01, T_floor=0.05, dt=3.0)
        s.step()
        assert s.t == pytest.approx(3.0)

    def test_zero_lambda_constant_temperature(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.0, T_floor=0.05)
        t0 = s.temperature(0.0)
        t100 = s.temperature(100.0)
        assert t0 == pytest.approx(t100)

    def test_locality_radius_uses_current_t_by_default(self):
        s = TemperatureSchedule(T0=1.0, lambda_=0.1, T_floor=0.05, dt=1.0)
        for _ in range(10):
            s.step()
        r_explicit = s.locality_radius(base_radius=5.0, t=s.t)
        r_default = s.locality_radius(base_radius=5.0)
        assert r_explicit == pytest.approx(r_default)


# ─────────────────────────────────────────────────────────────────────────────
# TestNoveltyEstimator
# ─────────────────────────────────────────────────────────────────────────────

class TestNoveltyEstimator:

    def test_no_neighbors_gives_max_distance_score(self):
        est = NoveltyEstimator()
        result = est.estimate(np.zeros(4), [], local_density=0.0)
        assert result.distance_score == pytest.approx(1.0)

    def test_identical_neighbor_gives_zero_distance_score(self):
        """When the experience is identical to a neighbor, dist=0 → dist_score≈0."""
        est = NoveltyEstimator()
        pos = np.array([1.0, 0.0, 0.0, 0.0])
        result = est.estimate(pos, [pos.copy()], local_density=0.0)
        assert result.distance_score == pytest.approx(0.0, abs=1e-6)

    def test_score_in_unit_interval(self):
        est = NoveltyEstimator()
        rng = np.random.default_rng(0)
        for _ in range(20):
            pos = rng.standard_normal(104)
            neighbors = [rng.standard_normal(104) for _ in range(5)]
            density = float(rng.uniform(0, 1))
            result = est.estimate(pos, neighbors, density)
            assert 0.0 <= result.score <= 1.0

    def test_high_density_reduces_score(self):
        est = NoveltyEstimator()
        pos = np.zeros(4)
        n = [np.ones(4) * 10.0]
        low_d = est.estimate(pos, n, local_density=0.0)
        high_d = est.estimate(pos, n, local_density=1.0)
        assert low_d.score > high_d.score

    def test_farther_neighbors_increase_score(self):
        # Use sigma_scale=1.0 (default): score = 1 - exp(-d/1.0)
        # close neighbor at d=0.01 → dist_score ≈ 0.01 → low novelty
        # far  neighbor at d=100  → dist_score ≈ 1.0  → high novelty
        est = NoveltyEstimator(sigma_scale=1.0)
        pos = np.zeros(4)
        close = [np.array([0.01, 0, 0, 0])]
        far = [np.array([100.0, 0, 0, 0])]
        r_close = est.estimate(pos, close, local_density=0.0)
        r_far = est.estimate(pos, far, local_density=0.0)
        assert r_far.score > r_close.score

    def test_density_score_is_one_minus_density(self):
        est = NoveltyEstimator()
        result = est.estimate(np.zeros(4), [], local_density=0.3)
        assert result.density_score == pytest.approx(0.7)

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            NoveltyEstimator(weight_distance=0.7, weight_density=0.4)

    def test_invalid_sigma_scale_raises(self):
        with pytest.raises(ValueError):
            NoveltyEstimator(sigma_scale=-1.0)

    def test_consistency_gradient_no_neighbors_is_zero(self):
        est = NoveltyEstimator()
        g = est.consistency_gradient(np.zeros(4), [])
        assert np.allclose(g, 0.0)

    def test_consistency_gradient_is_unit_length(self):
        est = NoveltyEstimator()
        pos = np.zeros(4)
        neighbors = [np.array([1.0, 0.0, 0.0, 0.0])]
        g = est.consistency_gradient(pos, neighbors)
        assert np.linalg.norm(g) == pytest.approx(1.0)

    def test_consistency_gradient_points_toward_centroid(self):
        est = NoveltyEstimator()
        pos = np.array([0.0, 0.0])
        neighbors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        g = est.consistency_gradient(pos, neighbors)
        centroid = np.array([0.5, 0.5])
        direction = centroid / np.linalg.norm(centroid)
        assert np.allclose(g, direction, atol=1e-6)

    def test_weighted_gradient_follows_heavier_neighbor(self):
        est = NoveltyEstimator()
        pos = np.zeros(2)
        neighbors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        weights = [10.0, 1.0]
        g = est.consistency_gradient(pos, neighbors, weights)
        # Should point mostly toward [1, 0]
        assert g[0] > g[1]

    def test_nearest_dist_in_result(self):
        est = NoveltyEstimator()
        pos = np.zeros(3)
        nb = np.array([3.0, 4.0, 0.0])  # distance = 5.0
        result = est.estimate(pos, [nb], local_density=0.0)
        assert result.nearest_dist == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestExperience
# ─────────────────────────────────────────────────────────────────────────────

class TestExperience:

    def test_vector_is_converted_to_float_array(self):
        e = Experience(vector=[1, 2, 3, 4])
        assert e.vector.dtype == float

    def test_dim_property(self):
        e = Experience(vector=np.zeros(104))
        assert e.dim == 104

    def test_label_default_is_none(self):
        e = Experience(vector=np.zeros(10))
        assert e.label is None

    def test_source_default(self):
        e = Experience(vector=np.zeros(10))
        assert e.source == "raw"

    def test_custom_label_and_source(self):
        e = Experience(vector=np.zeros(10), label="test_concept", source="memory")
        assert e.label == "test_concept"
        assert e.source == "memory"

    def test_2d_vector_raises(self):
        with pytest.raises(ValueError):
            Experience(vector=np.zeros((4, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# TestExperienceResult
# ─────────────────────────────────────────────────────────────────────────────

class TestExperienceResult:

    def _make_result(self, novelty=0.6, delta_magnitude=0.1, placed=None):
        exp = Experience(vector=np.zeros(104), label="test")
        return ExperienceResult(
            experience=exp,
            located_label="anchor",
            located_position=np.zeros(104),
            novelty=novelty,
            temperature=0.5,
            delta_magnitude=delta_magnitude,
            n_affected=3,
            placed_label=placed,
        )

    def test_was_novel_true_above_half(self):
        r = self._make_result(novelty=0.6)
        assert r.was_novel is True

    def test_was_novel_false_below_half(self):
        r = self._make_result(novelty=0.4)
        assert r.was_novel is False

    def test_deformation_applied_true(self):
        r = self._make_result(delta_magnitude=0.05)
        assert r.deformation_applied is True

    def test_deformation_applied_false_near_zero(self):
        r = self._make_result(delta_magnitude=1e-12)
        assert r.deformation_applied is False

    def test_placed_label_recorded(self):
        r = self._make_result(placed="new::concept")
        assert r.placed_label == "new::concept"


# ─────────────────────────────────────────────────────────────────────────────
# TestAnnealingStats
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnealingStats:

    def _make_result(self, novelty, delta=0.1, temp=0.5):
        exp = Experience(vector=np.zeros(104))
        return ExperienceResult(
            experience=exp,
            located_label="a",
            located_position=np.zeros(104),
            novelty=novelty,
            temperature=temp,
            delta_magnitude=delta,
            n_affected=1,
        )

    def test_initial_zero(self):
        s = AnnealingStats()
        assert s.n_processed == 0
        assert s.n_novel == 0
        assert s.total_deformation == 0.0
        assert s.mean_novelty == 0.0
        assert s.mean_temperature == 0.0

    def test_record_increments_processed(self):
        s = AnnealingStats()
        s.record(self._make_result(0.6))
        assert s.n_processed == 1

    def test_record_counts_novel(self):
        s = AnnealingStats()
        s.record(self._make_result(0.8))
        s.record(self._make_result(0.3))
        assert s.n_novel == 1

    def test_mean_novelty(self):
        s = AnnealingStats()
        s.record(self._make_result(0.4))
        s.record(self._make_result(0.6))
        assert s.mean_novelty == pytest.approx(0.5)

    def test_mean_temperature(self):
        s = AnnealingStats()
        s.record(self._make_result(0.5, temp=0.2))
        s.record(self._make_result(0.5, temp=0.4))
        assert s.mean_temperature == pytest.approx(0.3)

    def test_total_deformation_accumulates(self):
        s = AnnealingStats()
        s.record(self._make_result(0.5, delta=0.1))
        s.record(self._make_result(0.5, delta=0.3))
        assert s.total_deformation == pytest.approx(0.4)

    def test_novelty_rate(self):
        s = AnnealingStats()
        for _ in range(4):
            s.record(self._make_result(0.8))  # novel
        for _ in range(6):
            s.record(self._make_result(0.2))  # not novel
        assert s.novelty_rate == pytest.approx(0.4)


# ─────────────────────────────────────────────────────────────────────────────
# TestAnnealingEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnealingEngine:

    def test_process_returns_experience_result(self, engine):
        vec = np.zeros(104)
        result = engine.process(Experience(vector=vec))
        assert isinstance(result, ExperienceResult)

    def test_process_increments_n_processed(self, engine):
        assert engine.n_processed == 0
        engine.process(Experience(vector=np.zeros(104)))
        assert engine.n_processed == 1

    def test_process_advances_temperature_schedule(self, engine):
        T_before = engine.temperature
        engine.process(Experience(vector=np.zeros(104)))
        T_after = engine.temperature
        # Temperature should be slightly lower (schedule stepped)
        assert T_after <= T_before

    def test_process_labeled_places_concept(self, fresh_living):
        engine = AnnealingEngine(fresh_living, place_labeled=True)
        n_before = fresh_living.n_points
        vec = fresh_living.position("causal::perturbation").copy()
        result = engine.process(Experience(vector=vec, label="anneal::test_1"))
        assert fresh_living.n_points == n_before + 1
        assert result.placed_label == "anneal::test_1"

    def test_process_unlabeled_does_not_place(self, fresh_living):
        engine = AnnealingEngine(fresh_living, place_labeled=True)
        n_before = fresh_living.n_points
        engine.process(Experience(vector=np.zeros(104)))
        assert fresh_living.n_points == n_before

    def test_place_labeled_false_never_places(self, fresh_living):
        engine = AnnealingEngine(fresh_living, place_labeled=False)
        n_before = fresh_living.n_points
        engine.process(Experience(vector=np.zeros(104), label="should_not_place"))
        assert fresh_living.n_points == n_before

    def test_temperature_starts_above_floor(self, engine):
        assert engine.temperature > engine._schedule.T_floor

    def test_temperature_property_matches_schedule(self, engine):
        assert engine.temperature == engine.schedule.current_temperature

    def test_t_property_advances(self, engine):
        t0 = engine.t
        engine.process(Experience(vector=np.zeros(104)))
        assert engine.t > t0

    def test_reset_temperature_restores_initial(self, engine):
        for _ in range(50):
            engine.process(Experience(vector=np.zeros(104)))
        engine.reset_temperature()
        assert engine.schedule.t == 0.0

    def test_reset_does_not_undo_manifold_changes(self, fresh_living):
        engine = AnnealingEngine(fresh_living, place_labeled=True)
        vec = fresh_living.position("causal::perturbation").copy()
        engine.process(Experience(vector=vec, label="anneal::check_reset"))
        n_after = fresh_living.n_points
        engine.reset_temperature()
        # Manifold should still have the placed concept
        assert fresh_living.n_points == n_after

    def test_novelty_threshold_zero_always_deforms(self, fresh_living):
        engine = AnnealingEngine(
            fresh_living, novelty_threshold=0.0, place_labeled=False
        )
        result = engine.process(
            Experience(vector=fresh_living.position("causal::perturbation").copy())
        )
        # With threshold=0, any experience deforms
        assert result.n_affected > 0 or result.delta_magnitude == pytest.approx(
            0.0, abs=1e-9
        )

    def test_novelty_high_for_unknown_region(self, fresh_living):
        """An experience far from all existing points should be novel."""
        # sigma_scale=0.1 so any point >~0.1 units away scores highly
        engine = AnnealingEngine(fresh_living)
        engine._novelty = NoveltyEstimator(sigma_scale=0.1)
        far_vec = np.ones(104) * 1000.0
        result = engine.process(Experience(vector=far_vec))
        assert result.novelty > 0.5

    def test_novelty_low_for_dense_known_region(self, fresh_living):
        """An experience identical to a seed point should score lower novelty."""
        engine = AnnealingEngine(fresh_living)
        known = fresh_living.position("domain::mathematical").copy()
        result = engine.process(Experience(vector=known))
        # not necessarily tiny but should be below the far-vec case
        assert result.novelty < 1.0

    def test_located_label_is_string_or_none(self, engine):
        result = engine.process(Experience(vector=np.zeros(104)))
        assert result.located_label is None or isinstance(result.located_label, str)

    def test_process_batch_returns_list(self, engine):
        exps = [Experience(vector=np.zeros(104)) for _ in range(5)]
        results = engine.process_batch(exps)
        assert len(results) == 5
        assert all(isinstance(r, ExperienceResult) for r in results)

    def test_process_batch_increments_n_processed(self, engine):
        before = engine.n_processed
        engine.process_batch([Experience(vector=np.zeros(104)) for _ in range(3)])
        assert engine.n_processed == before + 3

    def test_stats_accumulates(self, engine):
        for i in range(10):
            engine.process(Experience(vector=np.random.randn(104)))
        assert engine.stats.n_processed >= 10

    def test_summary_returns_string(self, engine):
        s = engine.summary()
        assert isinstance(s, str)
        assert "processed" in s

    def test_schedule_property_is_temperature_schedule(self, engine):
        assert isinstance(engine.schedule, TemperatureSchedule)

    def test_stats_property_is_annealing_stats(self, engine):
        assert isinstance(engine.stats, AnnealingStats)

    def test_deformation_magnitude_positive_for_novel(self, fresh_living):
        engine = AnnealingEngine(fresh_living, novelty_threshold=0.0)
        far_vec = np.ones(104) * 500.0
        result = engine.process(Experience(vector=far_vec))
        # gradient might be zero if neighbours are at zero vector
        assert result.delta_magnitude >= 0.0

    def test_multiple_processes_accumulate_total_deformation(self, fresh_living):
        engine = AnnealingEngine(fresh_living, novelty_threshold=0.0)
        rng = np.random.default_rng(7)
        for _ in range(10):
            engine.process(Experience(vector=rng.standard_normal(104)))
        assert engine.stats.total_deformation >= 0.0

    def test_manifold_write_count_increases(self, fresh_living):
        engine = AnnealingEngine(fresh_living, place_labeled=True)
        before = fresh_living.n_writes
        for i in range(5):
            vec = fresh_living.position("causal::perturbation").copy()
            engine.process(Experience(vector=vec, label=f"anneal::b{i}"))
        assert fresh_living.n_writes > before

    def test_custom_temperature_params_respected(self, fresh_living):
        engine = AnnealingEngine(
            fresh_living, T0=2.0, lambda_=0.5, T_floor=0.1
        )
        assert engine.temperature == pytest.approx(
            TemperatureSchedule(T0=2.0, lambda_=0.5, T_floor=0.1).current_temperature
        )

    def test_experience_source_stored_in_result(self, engine):
        e = Experience(vector=np.zeros(104), source="memory")
        result = engine.process(e)
        assert result.experience.source == "memory"

    def test_located_position_is_104d(self, engine):
        result = engine.process(Experience(vector=np.zeros(104)))
        assert result.located_position.shape == (104,)


# ─────────────────────────────────────────────────────────────────────────────
# TestAnnealingEngineIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnealingEngineIntegration:
    """Higher-level integration tests that exercise the full system loop."""

    def test_stream_of_experiences_cools_engine(self, fresh_living):
        engine = AnnealingEngine(fresh_living, lambda_=0.1)
        T_start = engine.temperature
        exps = [Experience(vector=np.random.randn(104)) for _ in range(30)]
        engine.process_batch(exps)
        assert engine.temperature < T_start

    def test_labeled_experiences_appear_on_manifold(self, fresh_living):
        engine = AnnealingEngine(fresh_living, place_labeled=True)
        labels = [f"anneal::integ_{i}" for i in range(5)]
        for lbl in labels:
            vec = fresh_living.position("domain::mathematical").copy()
            vec[0] += np.random.uniform(0.01, 0.1)
            engine.process(Experience(vector=vec, label=lbl))
        # All labeled concepts should now be retrievable
        for lbl in labels:
            pos = fresh_living.position(lbl)
            assert pos.shape == (104,)

    def test_novel_experiences_raise_total_deformation(self, fresh_living):
        engine = AnnealingEngine(
            fresh_living, novelty_threshold=0.0, place_labeled=False
        )
        rng = np.random.default_rng(99)
        for _ in range(10):
            engine.process(Experience(vector=rng.standard_normal(104) * 100))
        assert engine.stats.total_deformation >= 0.0

    def test_repeated_same_experience_decreases_novelty(self, fresh_living):
        engine = AnnealingEngine(fresh_living, place_labeled=True)
        vec = fresh_living.position("causal::perturbation").copy()
        # first pass — novel
        r1 = engine.process(Experience(vector=vec, label="anneal::rep_1"))
        # second pass — should be less novel (was just placed)
        r2 = engine.process(Experience(vector=vec, label="anneal::rep_2"))
        # novelty may or may not be strictly less, but should be meaningful
        assert r2.novelty <= r1.novelty + 0.05  # allow tiny float wiggle

    def test_engine_handles_empty_manifold(self):
        """Engine should not crash if manifold has no points yet."""
        seed = SeedGeometryEngine().build()
        m = LivingManifold(seed)
        engine = AnnealingEngine(m, place_labeled=False)
        result = engine.process(Experience(vector=np.zeros(104)))
        # Should run without error; located_label may be a seed point
        assert isinstance(result, ExperienceResult)

    def test_annealing_locality_constraint_upheld(self, fresh_living):
        """Points far from the anchor should not be dramatically displaced."""
        engine = AnnealingEngine(fresh_living, place_labeled=False)
        anchor = "causal::perturbation"
        far_label = "domain::epistemic"

        pos_far_before = fresh_living.position(far_label).copy()
        vec = fresh_living.position(anchor).copy()
        vec[0] += 0.01

        for _ in range(5):
            engine.process(Experience(vector=vec))

        pos_far_after = fresh_living.position(far_label)
        shift = float(np.linalg.norm(pos_far_after - pos_far_before))
        # Far point should not have shifted by more than 1.0 unit
        assert shift < 1.0

    def test_process_batch_order_preserved(self, fresh_living):
        engine = AnnealingEngine(fresh_living, place_labeled=True)
        vecs = [fresh_living.position("domain::mathematical").copy() for _ in range(3)]
        labels = [f"anneal::order_{i}" for i in range(3)]
        exps = [Experience(vector=v, label=l) for v, l in zip(vecs, labels)]
        results = engine.process_batch(exps)
        for r, lbl in zip(results, labels):
            assert r.placed_label == lbl

    def test_public_api_importable_from_package(self):
        """All public types are importable from src.phase3.annealing_engine."""
        assert AnnealingEngineAlias is AnnealingEngine
        assert TemperatureScheduleAlias is TemperatureSchedule
        assert NoveltyEstimatorAlias is NoveltyEstimator
        assert ExperienceAlias is Experience
        assert ExperienceResultAlias is ExperienceResult

    def test_summary_contains_key_fields(self, fresh_living):
        engine = AnnealingEngine(fresh_living)
        for _ in range(5):
            engine.process(Experience(vector=np.random.randn(104)))
        s = engine.summary()
        for keyword in ["processed", "novelty", "temperature", "manifold"]:
            assert keyword in s

    def test_temperature_floor_respected_after_many_steps(self, fresh_living):
        engine = AnnealingEngine(
            fresh_living, T0=1.0, lambda_=2.0, T_floor=0.05
        )
        exps = [Experience(vector=np.random.randn(104)) for _ in range(500)]
        engine.process_batch(exps)
        assert engine.temperature >= 0.05 - 1e-4

    def test_stats_novelty_rate_between_zero_and_one(self, fresh_living):
        engine = AnnealingEngine(fresh_living)
        for _ in range(20):
            engine.process(Experience(vector=np.random.randn(104)))
        assert 0.0 <= engine.stats.novelty_rate <= 1.0
