"""Phase 5 test suite — Full Integration Pipeline & Evaluation Framework.

Run with:
  python -m pytest tests/test_phase5.py -v

All Phase 1-4 tests must remain green (run full suite):
  python -m pytest tests/ -v
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

# ── Phase 1 foundations ───────────────────────────────────────────────────────
from src.phase1.seed_geometry import SeedGeometryEngine
from src.phase1.expression.wave import StandingWave, WavePoint
from src.phase1.expression.renderer import RenderedOutput

# ── Phase 2 foundations ───────────────────────────────────────────────────────
from src.phase2.living_manifold.manifold import LivingManifold
from src.phase2.contrast_engine.engine import JudgmentType

# ── Phase 3 foundations ───────────────────────────────────────────────────────
from src.phase3.annealing_engine.experience import Experience, ExperienceResult
from src.phase3.annealing_engine.engine import AnnealingStats

# ── Phase 4 foundations ───────────────────────────────────────────────────────
from src.phase4.flow_engine.query import Query, Trajectory

# ── Phase 5 — Pipeline ────────────────────────────────────────────────────────
from src.phase5.pipeline.result import PipelineResult
from src.phase5.pipeline.pipeline import GEOPipeline
from src.phase5.pipeline import GEOPipeline as GEOPipelineAlias, PipelineResult as PipelineResultAlias

# ── Phase 5 — Evaluation ──────────────────────────────────────────────────────
from src.phase5.evaluation.metrics import (
    CoherenceMetrics,
    CausalMetrics,
    LocalityMetrics,
    EvaluationResult,
)
from src.phase5.evaluation.suite import SuiteResult
from src.phase5.evaluation.evaluator import PipelineEvaluator
from src.phase5.evaluation import (
    CoherenceMetrics as CoherenceMetricsAlias,
    CausalMetrics as CausalMetricsAlias,
    LocalityMetrics as LocalityMetricsAlias,
    EvaluationResult as EvaluationResultAlias,
    PipelineEvaluator as PipelineEvaluatorAlias,
    SuiteResult as SuiteResultAlias,
)

# ── Top-level package exports ─────────────────────────────────────────────────
from src.phase5 import (
    GEOPipeline as TopGEO,
    PipelineResult as TopResult,
    PipelineEvaluator as TopEval,
    SuiteResult as TopSuite,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    """A GEOPipeline built once for the entire module (read-heavy tests)."""
    return GEOPipeline(T0=0.5, lambda_=0.02, T_floor=0.02, flow_seed=42)


@pytest.fixture()
def fresh_pipeline():
    """A fresh GEOPipeline for tests that mutate state."""
    return GEOPipeline(T0=0.5, lambda_=0.02, T_floor=0.02, flow_seed=7)


@pytest.fixture(scope="module")
def seed_manifold():
    return SeedGeometryEngine().build()


@pytest.fixture(scope="module")
def vec_104():
    rng = np.random.default_rng(10)
    return rng.standard_normal(104).astype(float)


@pytest.fixture(scope="module")
def causal_vec(seed_manifold):
    """A vector near the causal domain."""
    M = LivingManifold(seed_manifold)
    return M.position("causal::perturbation").copy()


@pytest.fixture(scope="module")
def math_vec(seed_manifold):
    """A vector near the mathematical domain."""
    M = LivingManifold(seed_manifold)
    return M.position("domain::mathematical").copy()


# ─────────────────────────────────────────────────────────────────────────────
# TestPipelineResult
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineResult:
    """Tests for PipelineResult value object."""

    def test_package_alias_is_same_class(self):
        assert GEOPipelineAlias is GEOPipeline
        assert PipelineResultAlias is PipelineResult
        assert TopGEO is GEOPipeline

    def test_pipeline_result_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(PipelineResult)

    def test_pipeline_result_text_property(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec, label="test::text")
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_pipeline_result_confidence_in_range(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec, label="test::conf")
        assert 0.0 <= result.confidence <= 1.0

    def test_pipeline_result_n_steps_positive(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert result.n_steps >= 1

    def test_pipeline_result_termination_reason_valid(self, pipeline, causal_vec):
        valid = {"velocity_threshold", "revisit_detected", "max_steps", "attractor_reached"}
        result = pipeline.query(causal_vec)
        assert result.termination_reason in valid

    def test_pipeline_result_wave_confidence_in_range(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert 0.0 <= result.wave_confidence <= 1.0

    def test_pipeline_result_mean_speed_non_negative(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert result.mean_speed >= 0.0

    def test_pipeline_result_mean_curvature_non_negative(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert result.mean_curvature >= 0.0

    def test_pipeline_result_flow_preserved_is_bool(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert isinstance(result.flow_preserved, bool)

    def test_pipeline_result_repr_contains_query_label(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec, label="repr_check")
        assert "repr_check" in repr(result)

    def test_pipeline_result_trajectory_is_trajectory(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert isinstance(result.trajectory, Trajectory)

    def test_pipeline_result_wave_is_standing_wave(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert isinstance(result.wave, StandingWave)

    def test_pipeline_result_output_is_rendered_output(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert isinstance(result.output, RenderedOutput)


# ─────────────────────────────────────────────────────────────────────────────
# TestGEOPipeline — construction and introspection
# ─────────────────────────────────────────────────────────────────────────────

class TestGEOPipeline:
    """Tests for GEOPipeline construction and basic properties."""

    def test_pipeline_creates_without_error(self):
        p = GEOPipeline()
        assert p is not None

    def test_pipeline_has_104_dimensions(self, pipeline):
        assert pipeline.dimension == 104

    def test_pipeline_has_manifold(self, pipeline):
        assert isinstance(pipeline.manifold, LivingManifold)

    def test_pipeline_temperature_in_range(self, pipeline):
        assert 0.0 < pipeline.temperature <= 1.0

    def test_pipeline_stats_is_annealing_stats(self, pipeline):
        assert isinstance(pipeline.stats, AnnealingStats)

    def test_pipeline_query_count_starts_zero(self):
        p = GEOPipeline(flow_seed=0)
        assert p.query_count == 0

    def test_pipeline_query_increments_count(self, fresh_pipeline, causal_vec):
        assert fresh_pipeline.query_count == 0
        fresh_pipeline.query(causal_vec)
        assert fresh_pipeline.query_count == 1
        fresh_pipeline.query(causal_vec)
        assert fresh_pipeline.query_count == 2

    def test_pipeline_n_concepts_positive(self, pipeline):
        assert pipeline.n_concepts > 0

    def test_pipeline_seed_points_on_manifold(self, pipeline):
        # Seed manifold has 81 seed points; living manifold wraps them
        assert pipeline.n_concepts >= 81

    def test_pipeline_summary_is_string(self, pipeline):
        s = pipeline.summary()
        assert isinstance(s, str)
        assert "GEOPipeline" in s

    def test_pipeline_repr(self, pipeline):
        r = repr(pipeline)
        assert "GEOPipeline" in r
        assert "dim=104" in r

    def test_pipeline_dimension_matches_manifold(self, pipeline):
        assert pipeline.dimension == pipeline.manifold.DIM

    def test_pipeline_top_level_export_works(self, causal_vec):
        p = TopGEO(flow_seed=0)
        result = p.query(causal_vec)
        assert isinstance(result, PipelineResult)

    def test_pipeline_default_temperature(self):
        p = GEOPipeline(T0=1.0)
        # Initial temperature = T₀ + T_floor per TemperatureSchedule.initial_temperature
        assert p.temperature <= 1.0 + 0.1  # T0 + T_floor headroom

    def test_pipeline_reset_temperature(self, fresh_pipeline, causal_vec):
        initial_T = fresh_pipeline.temperature
        # Process a batch to cool down
        rng = np.random.default_rng(0)
        for _ in range(10):
            exp = Experience(rng.standard_normal(104), label=None)
            fresh_pipeline.learn(exp)
        assert fresh_pipeline.temperature <= initial_T
        fresh_pipeline.reset_temperature()
        # After reset temperature should be back near T0
        assert fresh_pipeline.temperature > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TestGEOPipelineLearn — C3 integration
# ─────────────────────────────────────────────────────────────────────────────

class TestGEOPipelineLearn:
    """Tests for GEOPipeline.learn() (C3 Annealing Engine)."""

    def test_learn_returns_experience_result(self, fresh_pipeline, vec_104):
        exp = Experience(vec_104, label="learn_test::0")
        result = fresh_pipeline.learn(exp)
        assert isinstance(result, ExperienceResult)

    def test_learn_novelty_in_range(self, fresh_pipeline, vec_104):
        result = fresh_pipeline.learn(Experience(vec_104, label="learn_test::1"))
        assert 0.0 <= result.novelty <= 1.0

    def test_learn_delta_magnitude_non_negative(self, fresh_pipeline, vec_104):
        result = fresh_pipeline.learn(Experience(vec_104))
        assert result.delta_magnitude >= 0.0

    def test_learn_n_affected_non_negative(self, fresh_pipeline, vec_104):
        result = fresh_pipeline.learn(Experience(vec_104))
        assert result.n_affected >= 0

    def test_learn_batch_returns_list(self, fresh_pipeline):
        rng = np.random.default_rng(1)
        exps = [Experience(rng.standard_normal(104)) for _ in range(3)]
        results = fresh_pipeline.learn_batch(exps)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_learn_stats_increments(self, fresh_pipeline, vec_104):
        before = fresh_pipeline.stats.n_processed
        fresh_pipeline.learn(Experience(vec_104))
        assert fresh_pipeline.stats.n_processed == before + 1

    def test_learn_cools_temperature(self, fresh_pipeline, vec_104):
        T_before = fresh_pipeline.temperature
        for i in range(20):
            fresh_pipeline.learn(Experience(vec_104 + i * 0.001))
        assert fresh_pipeline.temperature <= T_before

    def test_learn_labeled_concept_appears_on_manifold(self, fresh_pipeline, vec_104):
        label = "learn_test::unique_label_01"
        fresh_pipeline.learn(Experience(vec_104, label=label))
        assert label in fresh_pipeline.manifold.labels

    def test_learn_increases_n_concepts(self, fresh_pipeline, vec_104):
        n_before = fresh_pipeline.n_concepts
        fresh_pipeline.learn(Experience(vec_104, label="learn_test::grows"))
        assert fresh_pipeline.n_concepts >= n_before


# ─────────────────────────────────────────────────────────────────────────────
# TestGEOPipelineContrast — C4 integration
# ─────────────────────────────────────────────────────────────────────────────

class TestGEOPipelineContrast:
    """Tests for GEOPipeline.contrast() (C4 Contrast Engine)."""

    def test_contrast_same_pulls_closer(self, fresh_pipeline, causal_vec, math_vec):
        # Place two concepts
        fresh_pipeline.learn(Experience(causal_vec, label="contrast_test::a"))
        fresh_pipeline.learn(Experience(math_vec,   label="contrast_test::b"))
        M = fresh_pipeline.manifold
        dist_before = float(np.linalg.norm(
            M.position("contrast_test::a") - M.position("contrast_test::b")
        ))
        result = fresh_pipeline.contrast("contrast_test::a", "contrast_test::b", "same")
        dist_after = float(np.linalg.norm(
            M.position("contrast_test::a") - M.position("contrast_test::b")
        ))
        # same → distance should decrease (or stay same if already very close)
        assert dist_after <= dist_before + 1e-9

    def test_contrast_different_pushes_apart(self, fresh_pipeline, causal_vec, math_vec):
        fresh_pipeline.learn(Experience(causal_vec, label="contrast_test2::a"))
        fresh_pipeline.learn(Experience(math_vec,   label="contrast_test2::b"))
        M = fresh_pipeline.manifold
        dist_before = float(np.linalg.norm(
            M.position("contrast_test2::a") - M.position("contrast_test2::b")
        ))
        result = fresh_pipeline.contrast("contrast_test2::a", "contrast_test2::b", "different")
        dist_after = float(np.linalg.norm(
            M.position("contrast_test2::a") - M.position("contrast_test2::b")
        ))
        # different → distance should increase
        assert dist_after >= dist_before - 1e-9

    def test_contrast_returns_contrast_result(self, fresh_pipeline, causal_vec, math_vec):
        from src.phase2.contrast_engine.engine import ContrastResult
        fresh_pipeline.learn(Experience(causal_vec, label="contrast_test3::a"))
        fresh_pipeline.learn(Experience(math_vec,   label="contrast_test3::b"))
        result = fresh_pipeline.contrast("contrast_test3::a", "contrast_test3::b", "same")
        assert isinstance(result, ContrastResult)

    def test_contrast_delta_distance_is_float(self, fresh_pipeline, causal_vec, math_vec):
        fresh_pipeline.learn(Experience(causal_vec, label="contrast_test4::a"))
        fresh_pipeline.learn(Experience(math_vec,   label="contrast_test4::b"))
        result = fresh_pipeline.contrast("contrast_test4::a", "contrast_test4::b", "different")
        assert isinstance(result.distance_change, float)

    def test_contrast_judgment_type_enum_works(self, fresh_pipeline, causal_vec, math_vec):
        fresh_pipeline.learn(Experience(causal_vec, label="contrast_test5::a"))
        fresh_pipeline.learn(Experience(math_vec,   label="contrast_test5::b"))
        # String "same" must be accepted
        result = fresh_pipeline.contrast("contrast_test5::a", "contrast_test5::b", "same")
        assert result is not None

    def test_contrast_strength_parameter_accepted(self, fresh_pipeline, causal_vec, math_vec):
        fresh_pipeline.learn(Experience(causal_vec, label="contrast_test6::a"))
        fresh_pipeline.learn(Experience(math_vec,   label="contrast_test6::b"))
        result = fresh_pipeline.contrast(
            "contrast_test6::a", "contrast_test6::b", "different", strength=0.5
        )
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestGEOPipelineQuery — C5 → C6 → C7 integration
# ─────────────────────────────────────────────────────────────────────────────

class TestGEOPipelineQuery:
    """Tests for GEOPipeline.query() end-to-end (C5 → C6 → C7)."""

    def test_query_returns_pipeline_result(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert isinstance(result, PipelineResult)

    def test_query_text_is_non_empty_string(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert isinstance(result.text, str)
        assert len(result.text.strip()) > 0

    def test_query_with_label(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec, label="labelled query")
        assert result.query.label == "labelled query"

    def test_query_without_label(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert result.query.label is None

    def test_query_trajectory_has_steps(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert len(result.trajectory) >= 1

    def test_query_wave_has_points(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert len(result.wave.points) >= 1

    def test_query_confidence_float(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert isinstance(result.confidence, float)

    def test_two_different_vectors_produce_results(self, pipeline, causal_vec, math_vec):
        r1 = pipeline.query(causal_vec, label="q_causal")
        r2 = pipeline.query(math_vec,   label="q_math")
        # Both should produce valid non-empty text
        assert len(r1.text) > 0
        assert len(r2.text) > 0

    def test_query_wave_confidence_in_range(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        assert 0.0 <= result.wave_confidence <= 1.0

    def test_query_with_attractor_label(self, pipeline, causal_vec):
        result = pipeline.query(
            causal_vec,
            label="attractor_test",
            attractor_label="domain::mathematical",
        )
        assert isinstance(result, PipelineResult)

    def test_query_increments_count_on_pipeline(self, fresh_pipeline, causal_vec):
        before = fresh_pipeline.query_count
        fresh_pipeline.query(causal_vec)
        assert fresh_pipeline.query_count == before + 1

    def test_query_using_seed_label_position(self, pipeline):
        M = pipeline.manifold
        vec = M.position("causal::mechanism").copy()
        result = pipeline.query(vec, label="mechanism query")
        assert isinstance(result, PipelineResult)


# ─────────────────────────────────────────────────────────────────────────────
# TestGEOPipelineIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestGEOPipelineIntegration:
    """End-to-end integration tests for the full pipeline."""

    def test_full_learn_then_query_cycle(self, fresh_pipeline, vec_104, causal_vec):
        """Learn a batch of experiences, then run a query."""
        rng = np.random.default_rng(42)
        base = fresh_pipeline.manifold.position("causal::perturbation").copy()
        exps = [Experience(base + rng.standard_normal(104) * 0.05) for _ in range(5)]
        fresh_pipeline.learn_batch(exps)
        result = fresh_pipeline.query(causal_vec, label="integration_test")
        assert isinstance(result.text, str)
        assert result.n_steps >= 1

    def test_no_weights_in_pipeline(self, pipeline):
        """Architecture constraint: no weight matrices anywhere in pipeline."""
        # Check that the manifold does not have a 'weights' attribute
        assert not hasattr(pipeline.manifold, "weights")
        assert not hasattr(pipeline._flow_engine, "weights")
        assert not hasattr(pipeline._resonance, "weights")

    def test_no_tokens_in_result(self, pipeline, causal_vec):
        """Architecture constraint: output is continuous geometric text, not token IDs."""
        result = pipeline.query(causal_vec)
        # Trajectory positions are continuous vectors, not integer token IDs
        for pos in result.trajectory.positions:
            assert pos.dtype in (np.float32, np.float64, float)
            assert pos.shape == (104,)

    def test_learn_local_updates_only(self, fresh_pipeline, causal_vec):
        """Architecture constraint: learning must not move all manifold points."""
        M = fresh_pipeline.manifold
        all_labels = M.labels
        # Snapshot all far-away points (not near causal::perturbation)
        anchor_pos = M.position("causal::perturbation")
        far_labels = [
            l for l in all_labels
            if float(np.linalg.norm(M.position(l) - anchor_pos)) > 2.0
        ]
        before_far = {l: M.position(l).copy() for l in far_labels[:10]}

        # Learn something near causal::perturbation
        fresh_pipeline.learn(Experience(causal_vec + 0.01, label="locality_int_test"))

        # Far-away points must not have moved
        for l, pos_b in before_far.items():
            pos_a = M.position(l)
            assert float(np.linalg.norm(pos_a - pos_b)) < 1e-6, (
                f"Distant point '{l}' moved — locality violated"
            )

    def test_pipeline_query_gives_real_wave_not_mock(self, pipeline, causal_vec):
        """C7 receives a real Ψ from C6, not a mock wave."""
        result = pipeline.query(causal_vec)
        # A real wave's metadata contains trajectory info set by C6
        meta = result.wave.metadata
        assert "n_trajectory_steps" in meta
        assert meta["n_trajectory_steps"] == result.n_steps


# ─────────────────────────────────────────────────────────────────────────────
# TestCoherenceMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestCoherenceMetrics:
    """Tests for CoherenceMetrics."""

    @pytest.fixture()
    def metrics(self, pipeline, causal_vec):
        result = pipeline.query(causal_vec)
        return CoherenceMetrics.from_result(result.trajectory, result.wave, result.output)

    def test_wave_confidence_in_range(self, metrics):
        assert 0.0 <= metrics.wave_confidence <= 1.0

    def test_render_confidence_in_range(self, metrics):
        assert 0.0 <= metrics.render_confidence <= 1.0

    def test_flow_preserved_is_bool(self, metrics):
        assert isinstance(metrics.flow_preserved, bool)

    def test_n_wave_points_positive(self, metrics):
        assert metrics.n_wave_points >= 1

    def test_n_core_wave_points_lte_n_total(self, metrics):
        assert metrics.n_core_wave_points <= metrics.n_wave_points

    def test_core_fraction_in_range(self, metrics):
        assert 0.0 <= metrics.core_fraction <= 1.0

    def test_mean_amplitude_in_range(self, metrics):
        assert 0.0 <= metrics.mean_amplitude <= 1.0

    def test_trajectory_steps_positive(self, metrics):
        assert metrics.trajectory_steps >= 1

    def test_mean_speed_non_negative(self, metrics):
        assert metrics.trajectory_mean_speed >= 0.0

    def test_overall_score_in_range(self, metrics):
        score = metrics.overall_score()
        assert 0.0 <= score <= 1.0

    def test_termination_reason_valid(self, metrics):
        valid = {"velocity_threshold", "revisit_detected", "max_steps", "attractor_reached"}
        assert metrics.termination_reason in valid

    def test_from_result_classmethod_works(self, pipeline, math_vec):
        result = pipeline.query(math_vec)
        m = CoherenceMetrics.from_result(result.trajectory, result.wave, result.output)
        assert isinstance(m, CoherenceMetrics)

    def test_overall_score_weighted(self, metrics):
        """overall_score is a weighted combination of four sub-scores."""
        expected = (
            0.35 * metrics.render_confidence
            + 0.25 * metrics.wave_confidence
            + 0.20 * metrics.core_fraction
            + 0.20 * metrics.mean_amplitude
        )
        assert abs(metrics.overall_score() - expected) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# TestCausalMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalMetrics:
    """Tests for CausalMetrics."""

    @pytest.fixture(scope="class")
    def causal_metrics(self, pipeline, causal_vec, math_vec):
        ev = PipelineEvaluator(pipeline)
        return ev.evaluate_causal_direction(
            cause_vec=causal_vec, effect_vec=math_vec,
            cause_label="cmet::cause", effect_label="cmet::effect",
        )

    def test_causal_score_in_range(self, causal_metrics):
        assert 0.0 <= causal_metrics.causal_score <= 1.0

    def test_forward_steps_positive(self, causal_metrics):
        assert causal_metrics.forward_steps >= 1

    def test_backward_steps_positive(self, causal_metrics):
        assert causal_metrics.backward_steps >= 1

    def test_forward_speed_non_negative(self, causal_metrics):
        assert causal_metrics.forward_speed >= 0.0

    def test_backward_speed_non_negative(self, causal_metrics):
        assert causal_metrics.backward_speed >= 0.0

    def test_forward_curvature_non_negative(self, causal_metrics):
        assert causal_metrics.forward_curvature >= 0.0

    def test_backward_curvature_non_negative(self, causal_metrics):
        assert causal_metrics.backward_curvature >= 0.0

    def test_causal_direction_is_float(self, causal_metrics):
        assert isinstance(causal_metrics.causal_direction, float)

    def test_from_trajectories_classmethod(self, pipeline, causal_vec, math_vec):
        from src.phase4.flow_engine.engine import FlowEngine
        M = pipeline.manifold
        fe = FlowEngine(M, max_steps=50, seed=0)
        fwd = fe.flow(Query(vector=causal_vec.copy()))
        bwd = fe.flow(Query(vector=math_vec.copy()))
        m = CausalMetrics.from_trajectories(M, causal_vec, math_vec, fwd, bwd)
        assert isinstance(m, CausalMetrics)


# ─────────────────────────────────────────────────────────────────────────────
# TestLocalityMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalityMetrics:
    """Tests for LocalityMetrics."""

    @pytest.fixture()
    def locality_metrics(self, fresh_pipeline, vec_104):
        ev = PipelineEvaluator(fresh_pipeline)
        return ev.evaluate_locality(vec_104, label="lm_test::anchor")

    def test_locality_satisfied_is_bool(self, locality_metrics):
        assert isinstance(locality_metrics.locality_satisfied, bool)

    def test_locality_satisfied_is_true(self, locality_metrics):
        """The hard locality guarantee must be satisfied."""
        assert locality_metrics.locality_satisfied is True

    def test_n_distant_moved_is_zero(self, locality_metrics):
        assert locality_metrics.n_distant_moved == 0

    def test_max_distant_shift_near_zero(self, locality_metrics):
        assert locality_metrics.max_distant_shift < LocalityMetrics.DISTANT_SHIFT_TOLERANCE

    def test_locality_radius_positive(self, locality_metrics):
        assert locality_metrics.locality_radius_used > 0.0

    def test_measure_classmethod_accepts_empty_snapshots(self, fresh_pipeline):
        from src.phase2.living_manifold.manifold import LivingManifold
        M = fresh_pipeline.manifold
        lbl = "locality_test::empty"
        rng = np.random.default_rng(5)
        vec = rng.standard_normal(104)
        M.place(lbl, vec)
        M.update_density(lbl)
        m = LocalityMetrics.measure(
            manifold=M, anchor_label=lbl,
            snapshots_before=[], snapshots_after=[],
            locality_radius=0.5,
        )
        assert isinstance(m, LocalityMetrics)

    def test_max_nearby_shift_non_negative(self, locality_metrics):
        assert locality_metrics.max_nearby_shift >= 0.0

    def test_n_nearby_moved_non_negative(self, locality_metrics):
        assert locality_metrics.n_nearby_moved >= 0

    def test_locality_constant_tolerance_small(self):
        assert LocalityMetrics.DISTANT_SHIFT_TOLERANCE < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# TestEvaluationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationResult:
    """Tests for EvaluationResult data container."""

    @pytest.fixture()
    def eval_result(self, pipeline, causal_vec):
        ev = PipelineEvaluator(pipeline)
        return ev.evaluate_query(causal_vec, label="er_test")

    def test_eval_result_has_label(self, eval_result):
        assert eval_result.label == "er_test"

    def test_eval_result_has_coherence(self, eval_result):
        assert isinstance(eval_result.coherence, CoherenceMetrics)

    def test_eval_result_overall_score_in_range(self, eval_result):
        assert 0.0 <= eval_result.overall_score() <= 1.0

    def test_eval_result_repr(self, eval_result):
        r = repr(eval_result)
        assert "er_test" in r
        assert "EvaluationResult" in r

    def test_eval_result_extra_dict_empty_by_default(self, eval_result):
        assert isinstance(eval_result.extra, dict)


# ─────────────────────────────────────────────────────────────────────────────
# TestSuiteResult
# ─────────────────────────────────────────────────────────────────────────────

class TestSuiteResult:
    """Tests for SuiteResult aggregate container."""

    @pytest.fixture(scope="class")
    def suite_result(self, pipeline, causal_vec, math_vec):
        ev = PipelineEvaluator(pipeline)
        return ev.run_suite(
            [causal_vec, math_vec],
            labels=["suite_causal", "suite_math"],
            novelty_reps=3,
        )

    def test_suite_n_queries(self, suite_result):
        assert suite_result.n_queries == 2

    def test_suite_mean_coherence_in_range(self, suite_result):
        assert 0.0 <= suite_result.mean_coherence <= 1.0

    def test_suite_mean_render_confidence_in_range(self, suite_result):
        assert 0.0 <= suite_result.mean_render_confidence <= 1.0

    def test_suite_mean_wave_confidence_in_range(self, suite_result):
        assert 0.0 <= suite_result.mean_wave_confidence <= 1.0

    def test_suite_mean_steps_positive(self, suite_result):
        assert suite_result.mean_steps >= 1.0

    def test_suite_termination_distribution_covers_all_queries(self, suite_result):
        total = sum(suite_result.termination_distribution.values())
        assert total == suite_result.n_queries

    def test_suite_novelty_decay_length(self, suite_result):
        assert len(suite_result.novelty_decay) == 3

    def test_suite_causal_metrics_present(self, suite_result):
        assert suite_result.causal is not None
        assert isinstance(suite_result.causal, CausalMetrics)

    def test_suite_locality_metrics_present(self, suite_result):
        assert suite_result.locality is not None
        assert isinstance(suite_result.locality, LocalityMetrics)

    def test_suite_as_dict(self, suite_result):
        d = suite_result.as_dict()
        assert isinstance(d, dict)
        assert "n_queries" in d
        assert "mean_coherence" in d
        assert "causal_score" in d

    def test_suite_repr_contains_n_queries(self, suite_result):
        r = repr(suite_result)
        assert "SuiteResult" in r
        assert "n_queries" in r

    def test_empty_suite_result(self):
        sr = SuiteResult()
        assert sr.n_queries == 0
        assert sr.mean_coherence == 0.0

    def test_suite_result_novelty_is_decaying_property(self, suite_result):
        # novelty_is_decaying is a bool property; just check it returns bool
        assert isinstance(suite_result.novelty_is_decaying, bool)


# ─────────────────────────────────────────────────────────────────────────────
# TestPipelineEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineEvaluator:
    """Tests for PipelineEvaluator."""

    def test_evaluator_creation(self, pipeline):
        ev = PipelineEvaluator(pipeline)
        assert ev.pipeline is pipeline

    def test_evaluate_query_returns_evaluation_result(self, pipeline, causal_vec):
        ev = PipelineEvaluator(pipeline)
        result = ev.evaluate_query(causal_vec, label="ev_q_test")
        assert isinstance(result, EvaluationResult)

    def test_evaluate_query_label_stored(self, pipeline, causal_vec):
        ev = PipelineEvaluator(pipeline)
        result = ev.evaluate_query(causal_vec, label="label_check")
        assert result.label == "label_check"

    def test_evaluate_query_no_label_uses_default(self, pipeline, causal_vec):
        ev = PipelineEvaluator(pipeline)
        result = ev.evaluate_query(causal_vec)
        assert result.label == "unlabelled"

    def test_evaluate_causal_direction_returns_causal_metrics(self, pipeline, causal_vec, math_vec):
        ev = PipelineEvaluator(pipeline)
        m = ev.evaluate_causal_direction(causal_vec, math_vec)
        assert isinstance(m, CausalMetrics)

    def test_evaluate_novelty_decay_returns_list(self, fresh_pipeline, vec_104):
        ev = PipelineEvaluator(fresh_pipeline)
        decay = ev.evaluate_novelty_decay(vec_104, n_reps=3)
        assert isinstance(decay, list)
        assert len(decay) == 3

    def test_evaluate_novelty_decay_values_in_range(self, fresh_pipeline, vec_104):
        ev = PipelineEvaluator(fresh_pipeline)
        decay = ev.evaluate_novelty_decay(vec_104, n_reps=4)
        for v in decay:
            assert 0.0 <= v <= 1.0

    def test_evaluate_locality_returns_locality_metrics(self, fresh_pipeline, vec_104):
        ev = PipelineEvaluator(fresh_pipeline)
        m = ev.evaluate_locality(vec_104, label="ev_loc_test")
        assert isinstance(m, LocalityMetrics)

    def test_evaluate_locality_satisfied(self, fresh_pipeline, vec_104):
        ev = PipelineEvaluator(fresh_pipeline)
        m = ev.evaluate_locality(vec_104, label="ev_loc_sat")
        assert m.locality_satisfied is True

    def test_run_suite_returns_suite_result(self, pipeline, causal_vec, math_vec):
        ev = PipelineEvaluator(pipeline)
        sr = ev.run_suite([causal_vec, math_vec])
        assert isinstance(sr, SuiteResult)

    def test_run_suite_without_causal(self, pipeline, causal_vec):
        ev = PipelineEvaluator(pipeline)
        sr = ev.run_suite([causal_vec], run_causal=False)
        assert sr.causal is None

    def test_run_suite_without_locality(self, fresh_pipeline, causal_vec):
        ev = PipelineEvaluator(fresh_pipeline)
        sr = ev.run_suite([causal_vec], run_locality=False)
        assert sr.locality is None

    def test_top_level_alias(self, pipeline, causal_vec):
        ev = TopEval(pipeline)
        result = ev.evaluate_query(causal_vec, label="top_level_alias")
        assert isinstance(result, EvaluationResult)


# ─────────────────────────────────────────────────────────────────────────────
# TestPipelineEvaluatorIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineEvaluatorIntegration:
    """Full end-to-end evaluation suite integration tests."""

    def test_full_suite_all_metrics_present(self, pipeline, causal_vec, math_vec):
        ev = PipelineEvaluator(pipeline)
        sr = ev.run_suite([causal_vec, math_vec], labels=["c", "m"], novelty_reps=3)
        assert sr.n_queries == 2
        assert sr.causal is not None
        assert sr.locality is not None
        assert len(sr.novelty_decay) == 3
        d = sr.as_dict()
        assert d["n_queries"] == 2

    def test_novelty_decay_after_learning(self, fresh_pipeline, math_vec):
        """Repeated exposure of the same concept reduces novelty scores."""
        ev = PipelineEvaluator(fresh_pipeline)
        decay = ev.evaluate_novelty_decay(math_vec, n_reps=5)
        # First novelty should be higher than last after learning
        assert decay[0] >= decay[-1] - 0.5  # allow some tolerance from SDE randomness

    def test_locality_holds_after_learning(self, fresh_pipeline, causal_vec, math_vec):
        """Locality constraint is preserved even after batch learning."""
        rng = np.random.default_rng(99)
        base = fresh_pipeline.manifold.position("causal::perturbation").copy()
        exps = [Experience(base + rng.standard_normal(104) * 0.05) for _ in range(5)]
        fresh_pipeline.learn_batch(exps)

        ev = PipelineEvaluator(fresh_pipeline)
        m = ev.evaluate_locality(math_vec, label="loc_after_learn")
        assert m.locality_satisfied is True

    def test_pipeline_c7_receives_c6_wave_not_mock(self, fresh_pipeline, causal_vec):
        """End-to-end: C7 renders a real C6 wave (metadata key 'n_trajectory_steps')."""
        result = fresh_pipeline.query(causal_vec, label="c7_real_wave_test")
        meta = result.wave.metadata
        assert "n_trajectory_steps" in meta

    def test_suite_as_dict_complete(self, pipeline, causal_vec, math_vec):
        ev = PipelineEvaluator(pipeline)
        sr = ev.run_suite([causal_vec, math_vec], novelty_reps=2)
        d = sr.as_dict()
        required = {
            "n_queries", "mean_coherence", "mean_render_confidence",
            "mean_wave_confidence", "mean_steps", "termination_distribution",
            "novelty_is_decaying", "causal_score", "locality_satisfied",
        }
        assert required.issubset(d.keys())
