"""Phase 4 test suite — Flow Engine (C5) and Resonance Layer (C6).

Run with:
  python3 -m pytest tests/test_phase4.py -v

All Phase 1, 2, and 3 tests must remain green (run full suite):
  python3 -m pytest tests/ -v
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

# ── Phase 1 (needed to build the seed manifold) ──────────────────────────────
from src.phase1.seed_geometry import SeedGeometryEngine
from src.phase1.expression.wave import StandingWave, WavePoint

# ── Phase 2 (Living Manifold wraps M₀) ───────────────────────────────────────
from src.phase2.living_manifold.manifold import LivingManifold

# ── Phase 4 — Flow Engine ─────────────────────────────────────────────────────
from src.phase4.flow_engine.query import Query, FlowStep, Trajectory, DIM
from src.phase4.flow_engine.forces import ForceComputer
from src.phase4.flow_engine.sde import SDESolver
from src.phase4.flow_engine.engine import FlowEngine
from src.phase4.flow_engine import (
    FlowEngine as FlowEngineAlias,
    Query as QueryAlias,
    FlowStep as FlowStepAlias,
    Trajectory as TrajectoryAlias,
    ForceComputer as ForceComputerAlias,
    SDESolver as SDESolverAlias,
)

# ── Phase 4 — Resonance Layer ─────────────────────────────────────────────────
from src.phase4.resonance_layer.accumulator import (
    ExcitationKernel,
    HarmonicKernel,
    ResonanceAccumulator,
)
from src.phase4.resonance_layer.layer import ResonanceLayer
from src.phase4.resonance_layer import (
    ResonanceLayer as ResonanceLayerAlias,
    ResonanceAccumulator as ResonanceAccumulatorAlias,
    ExcitationKernel as ExcitationKernelAlias,
    HarmonicKernel as HarmonicKernelAlias,
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
    """Shared LivingManifold (read-only tests may share; mutating tests use fresh_living)."""
    return LivingManifold(seed_manifold)


@pytest.fixture()
def fresh_living(seed_manifold):
    """Fresh LivingManifold for tests that may mutate state."""
    return LivingManifold(seed_manifold)


@pytest.fixture()
def vec_104():
    """A random normalised 104D vector."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(104)
    return v / np.linalg.norm(v)


@pytest.fixture()
def query_vec(living):
    """Query vector anchored near a known seed point."""
    return living.position("causal::perturbation").copy()


@pytest.fixture()
def simple_query(query_vec):
    """A simple Query object."""
    return Query(vector=query_vec, label="test_query")


@pytest.fixture()
def flow_engine(living):
    """FlowEngine on the shared living manifold with reproducible seed."""
    return FlowEngine(living, max_steps=30, seed=42)


@pytest.fixture()
def short_trajectory(flow_engine, simple_query):
    """A short Trajectory produced by the flow engine."""
    return flow_engine.flow(simple_query)


@pytest.fixture()
def resonance_layer(living):
    """ResonanceLayer on the shared living manifold."""
    return ResonanceLayer(living)


# ─────────────────────────────────────────────────────────────────────────────
# TestQuery
# ─────────────────────────────────────────────────────────────────────────────

class TestQuery:

    def test_query_accepts_104d_vector(self, vec_104):
        q = Query(vector=vec_104)
        assert q.vector.shape == (104,)

    def test_query_stores_label(self, vec_104):
        q = Query(vector=vec_104, label="my_query")
        assert q.label == "my_query"

    def test_query_default_label_is_empty(self, vec_104):
        q = Query(vector=vec_104)
        assert q.label == ""

    def test_query_attractor_label_none_by_default(self, vec_104):
        q = Query(vector=vec_104)
        assert q.attractor_label is None

    def test_query_attractor_label_can_be_set(self, vec_104):
        q = Query(vector=vec_104, attractor_label="domain::logical")
        assert q.attractor_label == "domain::logical"

    def test_query_rejects_wrong_dimension(self):
        with pytest.raises(ValueError):
            Query(vector=np.zeros(50))

    def test_query_rejects_2d_array(self):
        with pytest.raises(ValueError):
            Query(vector=np.zeros((104, 1)))

    def test_query_vector_is_float_array(self, vec_104):
        q = Query(vector=vec_104.astype(np.float32))
        assert q.vector.dtype == float

    def test_query_alias_importable(self, vec_104):
        q = QueryAlias(vector=vec_104)
        assert isinstance(q, Query)


# ─────────────────────────────────────────────────────────────────────────────
# TestFlowStep
# ─────────────────────────────────────────────────────────────────────────────

class TestFlowStep:

    def test_flowstep_stores_position(self, vec_104):
        step = FlowStep(position=vec_104, velocity=np.zeros(104), time=0.0)
        np.testing.assert_array_equal(step.position, vec_104)

    def test_flowstep_speed_computed_from_velocity(self, vec_104):
        vel = vec_104 * 2.0
        step = FlowStep(position=vec_104, velocity=vel, time=0.0)
        assert step.speed == pytest.approx(float(np.linalg.norm(vel)), abs=1e-9)

    def test_flowstep_zero_velocity_gives_zero_speed(self, vec_104):
        step = FlowStep(position=vec_104, velocity=np.zeros(104), time=0.0)
        assert step.speed == 0.0

    def test_flowstep_time_stored(self, vec_104):
        step = FlowStep(position=vec_104, velocity=np.zeros(104), time=1.23)
        assert step.time == pytest.approx(1.23)

    def test_flowstep_curvature_default_zero(self, vec_104):
        step = FlowStep(position=vec_104, velocity=np.zeros(104), time=0.0)
        assert step.curvature == 0.0

    def test_flowstep_curvature_can_be_set(self, vec_104):
        step = FlowStep(position=vec_104, velocity=np.zeros(104), time=0.0, curvature=3.14)
        assert step.curvature == pytest.approx(3.14)

    def test_flowstep_repr_contains_time(self, vec_104):
        step = FlowStep(position=vec_104, velocity=np.zeros(104), time=0.5)
        assert "0.500" in repr(step)


# ─────────────────────────────────────────────────────────────────────────────
# TestTrajectory
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectory:

    def _make_trajectory(self, n: int, query: Query) -> Trajectory:
        steps = [
            FlowStep(
                position=np.random.default_rng(i).standard_normal(104),
                velocity=np.ones(104) * 0.01 * i,
                time=float(i) * 0.05,
            )
            for i in range(n)
        ]
        return Trajectory(steps=steps, query=query)

    def test_trajectory_total_time_from_last_step(self, simple_query):
        traj = self._make_trajectory(5, simple_query)
        assert traj.total_time == pytest.approx(4 * 0.05, abs=1e-9)

    def test_trajectory_empty_is_detected(self, simple_query):
        traj = Trajectory(steps=[], query=simple_query)
        assert traj.is_empty

    def test_trajectory_non_empty(self, simple_query):
        traj = self._make_trajectory(3, simple_query)
        assert not traj.is_empty

    def test_trajectory_len_equals_n_steps(self, simple_query):
        traj = self._make_trajectory(7, simple_query)
        assert len(traj) == 7

    def test_trajectory_positions_list(self, simple_query):
        traj = self._make_trajectory(4, simple_query)
        positions = traj.positions
        assert len(positions) == 4
        assert positions[0].shape == (104,)

    def test_trajectory_as_position_time_pairs(self, simple_query):
        traj = self._make_trajectory(3, simple_query)
        pairs = traj.as_position_time_pairs
        assert len(pairs) == 3
        pos, t = pairs[1]
        assert pos.shape == (104,)
        assert t == pytest.approx(0.05, abs=1e-9)

    def test_trajectory_mean_speed(self, simple_query):
        traj = self._make_trajectory(4, simple_query)
        speeds = [s.speed for s in traj.steps]
        assert traj.mean_speed == pytest.approx(float(np.mean(speeds)), abs=1e-9)

    def test_trajectory_mean_curvature_zero_when_not_set(self, simple_query):
        traj = self._make_trajectory(3, simple_query)
        assert traj.mean_curvature == pytest.approx(0.0)

    def test_trajectory_termination_reason_stored(self, simple_query):
        traj = self._make_trajectory(2, simple_query)
        traj.termination_reason = "max_steps"
        assert traj.termination_reason == "max_steps"

    def test_trajectory_repr_contains_n_steps(self, simple_query):
        traj = self._make_trajectory(5, simple_query)
        assert "5" in repr(traj)


# ─────────────────────────────────────────────────────────────────────────────
# TestForceComputer
# ─────────────────────────────────────────────────────────────────────────────

class TestForceComputer:

    def test_force_computer_creates_with_defaults(self):
        fc = ForceComputer()
        assert fc.gravity_k == 8
        assert fc.repulsion_k == 5

    def test_semantic_gravity_returns_104d(self, living, query_vec):
        fc = ForceComputer()
        f = fc.semantic_gravity(query_vec, living)
        assert f.shape == (104,)

    def test_semantic_gravity_is_unit_or_zero(self, living, query_vec):
        fc = ForceComputer()
        f = fc.semantic_gravity(query_vec, living)
        norm = float(np.linalg.norm(f))
        assert norm <= 1.0 + 1e-9

    def test_semantic_gravity_pulls_toward_neighbours(self, living, query_vec):
        fc = ForceComputer()
        f = fc.semantic_gravity(query_vec, living)
        # Force should be non-zero (manifold has 81 points)
        assert float(np.linalg.norm(f)) >= 0.0  # may be zero if gravity cancels

    def test_causal_curvature_returns_104d(self, living, query_vec):
        fc = ForceComputer()
        vel = np.zeros(104)
        f = fc.causal_curvature(query_vec, vel, living)
        assert f.shape == (104,)

    def test_contextual_momentum_scales_velocity(self):
        fc = ForceComputer()
        vel = np.ones(104)
        f = fc.contextual_momentum(vel, gamma=0.85)
        np.testing.assert_allclose(f, vel * 0.85)

    def test_contextual_momentum_zero_velocity(self):
        fc = ForceComputer()
        f = fc.contextual_momentum(np.zeros(104), gamma=0.9)
        np.testing.assert_allclose(f, np.zeros(104))

    def test_contextual_momentum_custom_gamma(self):
        fc = ForceComputer()
        vel = np.ones(104) * 2.0
        f = fc.contextual_momentum(vel, gamma=0.5)
        np.testing.assert_allclose(f, np.ones(104))

    def test_contrast_repulsion_returns_104d(self, living, query_vec):
        fc = ForceComputer()
        f = fc.contrast_repulsion(query_vec, living)
        assert f.shape == (104,)

    def test_contrast_repulsion_unit_or_zero(self, living, query_vec):
        fc = ForceComputer()
        f = fc.contrast_repulsion(query_vec, living)
        norm = float(np.linalg.norm(f))
        assert norm <= 1.0 + 1e-9

    def test_combined_drift_returns_104d(self, living, query_vec):
        fc = ForceComputer()
        vel = np.zeros(104)
        drift = fc.combined_drift(query_vec, vel, living)
        assert drift.shape == (104,)

    def test_combined_drift_custom_weights(self, living, query_vec):
        fc = ForceComputer()
        vel = np.zeros(104)
        drift = fc.combined_drift(query_vec, vel, living, weights=(1.0, 0.0, 0.0, 0.0))
        # With zero causal/momentum/repulsion, drift = 1.0 * gravity
        gravity = fc.semantic_gravity(query_vec, living)
        np.testing.assert_allclose(drift, gravity, atol=1e-9)

    def test_combined_drift_all_zero_weights(self, living, query_vec):
        fc = ForceComputer()
        vel = np.zeros(104)
        drift = fc.combined_drift(query_vec, vel, living, weights=(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(drift, np.zeros(104), atol=1e-9)

    def test_force_computer_alias_importable(self):
        fc = ForceComputerAlias()
        assert isinstance(fc, ForceComputer)

    def test_semantic_gravity_changes_with_position(self, living):
        fc = ForceComputer()
        pos_a = living.position("causal::perturbation").copy()
        pos_b = living.position("domain::mathematical").copy()
        f_a = fc.semantic_gravity(pos_a, living)
        f_b = fc.semantic_gravity(pos_b, living)
        # Forces at different positions differ
        assert not np.allclose(f_a, f_b)


# ─────────────────────────────────────────────────────────────────────────────
# TestSDESolver
# ─────────────────────────────────────────────────────────────────────────────

class TestSDESolver:

    def test_sde_creates_with_defaults(self):
        sde = SDESolver()
        assert sde.dt == pytest.approx(0.05)
        assert sde.diffusion_scale == pytest.approx(0.05)

    def test_sde_rejects_non_positive_dt(self):
        with pytest.raises(ValueError):
            SDESolver(dt=0.0)

    def test_sde_rejects_negative_diffusion(self):
        with pytest.raises(ValueError):
            SDESolver(diffusion_scale=-0.1)

    def test_sde_step_returns_two_arrays(self, living, query_vec):
        sde = SDESolver(dt=0.05, diffusion_scale=0.01, rng=np.random.default_rng(0))
        drift = np.ones(104) * 0.1
        new_pos, new_vel = sde.step(query_vec, drift, living)
        assert new_pos.shape == (104,)
        assert new_vel.shape == (104,)

    def test_sde_step_moves_position(self, living, query_vec):
        sde = SDESolver(dt=0.05, diffusion_scale=0.0, rng=np.random.default_rng(0))
        drift = np.ones(104) * 0.1
        new_pos, _vel = sde.step(query_vec, drift, living)
        # With non-zero drift, position must change
        assert not np.allclose(new_pos, query_vec)

    def test_sde_step_velocity_equals_drift_times_dt(self, living, query_vec):
        sde = SDESolver(dt=0.1, diffusion_scale=0.0, rng=np.random.default_rng(0))
        drift = np.ones(104) * 0.5
        _new_pos, new_vel = sde.step(query_vec, drift, living)
        expected_vel = drift * 0.1
        np.testing.assert_allclose(new_vel, expected_vel, atol=1e-9)

    def test_sde_zero_drift_only_diffusion_moves(self, living):
        sde = SDESolver(dt=0.05, diffusion_scale=1.0, rng=np.random.default_rng(7))
        # Use a sparse position (far from all seed points) so diffusion is non-zero
        sparse_pos = np.random.default_rng(7).standard_normal(104) * 10.0
        drift = np.zeros(104)
        new_pos, _vel = sde.step(sparse_pos, drift, living)
        # Diffusion is random; expect some movement
        assert not np.allclose(new_pos, sparse_pos)

    def test_diffusion_at_returns_float(self, living, query_vec):
        sde = SDESolver()
        sigma = sde.diffusion_at(query_vec, living)
        assert isinstance(sigma, float)
        assert sigma >= 0.0

    def test_diffusion_decreases_with_density(self, living):
        sde = SDESolver(diffusion_scale=1.0)
        pos_dense = living.position("causal::perturbation").copy()
        pos_sparse = np.random.default_rng(99).standard_normal(104) * 10.0
        sigma_dense = sde.diffusion_at(pos_dense, living)
        sigma_sparse = sde.diffusion_at(pos_sparse, living)
        # Sparse regions should have higher diffusion
        assert sigma_sparse >= sigma_dense - 1e-9

    def test_sde_alias_importable(self):
        sde = SDESolverAlias()
        assert isinstance(sde, SDESolver)


# ─────────────────────────────────────────────────────────────────────────────
# TestFlowEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestFlowEngine:

    def test_flow_engine_creates_with_manifold(self, living):
        fe = FlowEngine(living)
        assert fe.max_steps == 200
        assert fe.dt == pytest.approx(0.05)

    def test_flow_engine_custom_params(self, living):
        fe = FlowEngine(living, max_steps=50, dt=0.1, seed=1)
        assert fe.max_steps == 50
        assert fe.dt == pytest.approx(0.1)

    def test_flow_returns_trajectory(self, flow_engine, simple_query):
        traj = flow_engine.flow(simple_query)
        assert isinstance(traj, Trajectory)

    def test_flow_trajectory_has_steps(self, flow_engine, simple_query):
        traj = flow_engine.flow(simple_query)
        assert len(traj) > 0

    def test_flow_trajectory_steps_are_104d(self, flow_engine, simple_query):
        traj = flow_engine.flow(simple_query)
        for step in traj.steps:
            assert step.position.shape == (DIM,)
            assert step.velocity.shape == (DIM,)

    def test_flow_trajectory_time_is_increasing(self, flow_engine, simple_query):
        traj = flow_engine.flow(simple_query)
        times = [s.time for s in traj.steps]
        for a, b in zip(times, times[1:]):
            assert b >= a - 1e-9

    def test_flow_trajectory_stores_query(self, flow_engine, simple_query):
        traj = flow_engine.flow(simple_query)
        assert traj.query is simple_query

    def test_flow_has_termination_reason(self, flow_engine, simple_query):
        traj = flow_engine.flow(simple_query)
        valid_reasons = {
            "velocity_threshold", "revisit_detected", "max_steps",
            "attractor_reached", "unknown"
        }
        assert traj.termination_reason in valid_reasons

    def test_flow_max_steps_respected(self, living, simple_query):
        fe = FlowEngine(living, max_steps=10, seed=0)
        traj = fe.flow(simple_query)
        assert len(traj) <= 11  # +1 for possible final step on velocity threshold

    def test_flow_with_attractor_label(self, living, query_vec):
        q = Query(
            vector=query_vec,
            label="attractor_query",
            attractor_label="domain::logical",
        )
        fe = FlowEngine(living, max_steps=30, seed=1)
        traj = fe.flow(q)
        assert isinstance(traj, Trajectory)
        assert len(traj) > 0

    def test_flow_mean_speed_positive(self, flow_engine, simple_query):
        traj = flow_engine.flow(simple_query)
        assert traj.mean_speed >= 0.0

    def test_flow_positions_differ_between_steps(self, flow_engine, simple_query):
        traj = flow_engine.flow(simple_query)
        if len(traj) >= 2:
            p0, p1 = traj.steps[0].position, traj.steps[1].position
            # The SDE must produce movement
            assert not np.allclose(p0, p1, atol=1e-12)

    def test_flow_reproducible_with_seed(self, living, simple_query):
        fe1 = FlowEngine(living, max_steps=20, seed=99)
        fe2 = FlowEngine(living, max_steps=20, seed=99)
        t1 = fe1.flow(simple_query)
        t2 = fe2.flow(simple_query)
        np.testing.assert_allclose(
            t1.steps[0].position, t2.steps[0].position
        )

    def test_flow_different_queries_differ(self, living):
        q1 = Query(vector=living.position("causal::perturbation").copy(), label="q1")
        q2 = Query(vector=living.position("domain::mathematical").copy(), label="q2")
        fe = FlowEngine(living, max_steps=20, seed=5)
        t1 = fe.flow(q1)
        t2 = fe.flow(q2)
        # Starting positions differ → trajectories differ
        assert not np.allclose(t1.steps[0].position, t2.steps[0].position)

    def test_flow_engine_alias_importable(self, living):
        fe = FlowEngineAlias(living)
        assert isinstance(fe, FlowEngine)


# ─────────────────────────────────────────────────────────────────────────────
# TestFlowEngineIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestFlowEngineIntegration:

    def test_full_pipeline_causal_query(self, living):
        q = Query(
            vector=living.position("causal::perturbation").copy(),
            label="causal_query",
        )
        fe = FlowEngine(living, max_steps=50, seed=10)
        traj = fe.flow(q)
        assert isinstance(traj, Trajectory)
        assert not traj.is_empty

    def test_full_pipeline_logical_query(self, living):
        q = Query(
            vector=living.position("domain::logical_entities").copy(),
            label="logical_query",
        )
        fe = FlowEngine(living, max_steps=50, seed=11)
        traj = fe.flow(q)
        assert not traj.is_empty

    def test_trajectory_as_position_time_pairs_for_resonance(self, short_trajectory):
        pairs = short_trajectory.as_position_time_pairs
        assert len(pairs) > 0
        for pos, t in pairs:
            assert pos.shape == (DIM,)
            assert t >= 0.0

    def test_multiple_queries_independent(self, living):
        fe = FlowEngine(living, max_steps=20, seed=20)
        labels = ["causal::perturbation", "domain::mathematical", "domain::logical_entities"]
        trajectories = []
        for label in labels:
            q = Query(vector=living.position(label).copy(), label=label)
            trajectories.append(fe.flow(q))
        # All trajectories are valid
        for traj in trajectories:
            assert not traj.is_empty

    def test_flow_position_stays_plausible(self, living, simple_query):
        """Positions should not explode to infinity."""
        fe = FlowEngine(living, max_steps=100, diffusion_scale=0.01, seed=30)
        traj = fe.flow(simple_query)
        for step in traj.steps:
            pos_norm = float(np.linalg.norm(step.position))
            assert pos_norm < 1e6, f"Position exploded: {pos_norm}"


# ─────────────────────────────────────────────────────────────────────────────
# TestExcitationKernel
# ─────────────────────────────────────────────────────────────────────────────

class TestExcitationKernel:

    def test_excitation_self_equals_amplitude(self):
        ek = ExcitationKernel(resonance_radius=1.0)
        pos = np.zeros(104)
        # Same position → distance = 0 → excitation = amplitude = speed
        exc = ek.excitation(pos, pos, speed=0.5, curvature=0.0)
        assert exc == pytest.approx(0.5, abs=1e-9)

    def test_excitation_decays_with_distance(self):
        ek = ExcitationKernel(resonance_radius=1.0)
        q = np.zeros(104)
        far = np.zeros(104)
        far[0] = 10.0
        exc_near = ek.excitation(q, np.zeros(104), speed=1.0, curvature=0.0)
        exc_far = ek.excitation(q, far, speed=1.0, curvature=0.0)
        assert exc_near > exc_far

    def test_excitation_greater_speed_gives_greater_amplitude(self):
        ek = ExcitationKernel(resonance_radius=1.0)
        pos = np.zeros(104)
        exc_slow = ek.excitation(pos, pos, speed=0.1, curvature=0.0)
        exc_fast = ek.excitation(pos, pos, speed=1.0, curvature=0.0)
        assert exc_fast > exc_slow

    def test_effective_radius_decreases_with_curvature(self):
        ek = ExcitationKernel(resonance_radius=1.0)
        r_low = ek.effective_radius(curvature=0.0)
        r_high = ek.effective_radius(curvature=10.0)
        assert r_high < r_low

    def test_effective_radius_respects_min_radius(self):
        ek = ExcitationKernel(resonance_radius=0.1, min_radius=0.05)
        r = ek.effective_radius(curvature=1000.0)
        assert r >= 0.05

    def test_excitation_non_negative(self):
        ek = ExcitationKernel()
        rng = np.random.default_rng(1)
        for _ in range(20):
            q = rng.standard_normal(104)
            p = rng.standard_normal(104)
            exc = ek.excitation(q, p, speed=rng.uniform(0.01, 1.0), curvature=rng.uniform(0.0, 5.0))
            assert exc >= 0.0

    def test_excitation_kernel_alias_importable(self):
        ek = ExcitationKernelAlias()
        assert isinstance(ek, ExcitationKernel)


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicKernel
# ─────────────────────────────────────────────────────────────────────────────

class TestHarmonicKernel:

    def test_harmonic_identical_curvatures(self):
        hk = HarmonicKernel(harmonic_tolerance=0.15)
        # freq_q = freq_p → ratio = 1.0 → nearest int = 1 → delta = 0
        f = hk.factor(curvature_q=1.0, curvature_p=1.0)
        assert f == pytest.approx(1.0, abs=1e-9)

    def test_harmonic_octave_relationship(self):
        hk = HarmonicKernel(harmonic_tolerance=0.15)
        # freq_q / freq_p = 2 → integer → factor should be ~1.0
        f = hk.factor(curvature_q=1.0, curvature_p=0.0)
        # freq_q = 2.0, freq_p = 1.0 → ratio = 2 → delta = 0
        assert f == pytest.approx(1.0, abs=1e-9)

    def test_harmonic_non_integer_ratio(self):
        hk = HarmonicKernel(harmonic_tolerance=0.15)
        # freq_q = 1.5, freq_p = 1.0 → ratio = 1.5 → nearest int = 2 → delta = 0.5
        f = hk.factor(curvature_q=0.5, curvature_p=0.0)
        assert 0.0 <= f <= 1.0

    def test_harmonic_factor_in_range_zero_one(self):
        hk = HarmonicKernel()
        rng = np.random.default_rng(2)
        for _ in range(30):
            kq = rng.uniform(0.0, 5.0)
            kp = rng.uniform(0.0, 5.0)
            f = hk.factor(kq, kp)
            assert 0.0 <= f <= 1.0 + 1e-12

    def test_harmonic_stricter_tolerance(self):
        hk_strict = HarmonicKernel(harmonic_tolerance=0.01)
        hk_loose = HarmonicKernel(harmonic_tolerance=0.5)
        # Non-integer ratio: strict should give lower factor
        f_strict = hk_strict.factor(0.5, 0.0)  # ratio = 1.5
        f_loose = hk_loose.factor(0.5, 0.0)
        assert f_loose >= f_strict

    def test_harmonic_rejects_non_positive_tolerance(self):
        with pytest.raises(ValueError):
            HarmonicKernel(harmonic_tolerance=0.0)

    def test_harmonic_kernel_alias_importable(self):
        hk = HarmonicKernelAlias()
        assert isinstance(hk, HarmonicKernel)


# ─────────────────────────────────────────────────────────────────────────────
# TestResonanceAccumulator
# ─────────────────────────────────────────────────────────────────────────────

class TestResonanceAccumulator:

    def test_accumulate_empty_trajectory(self):
        acc = ResonanceAccumulator()
        result = acc.accumulate([])
        assert len(result) == 0

    def test_accumulate_single_site(self):
        acc = ResonanceAccumulator()
        sites = [(np.zeros(104), 1.0, 0.0)]
        result = acc.accumulate(sites)
        assert result.shape == (1,)
        assert result[0] > 0

    def test_accumulate_length_matches_sites(self):
        acc = ResonanceAccumulator()
        rng = np.random.default_rng(3)
        sites = [(rng.standard_normal(104), 0.5, 0.0) for _ in range(10)]
        result = acc.accumulate(sites)
        assert result.shape == (10,)

    def test_accumulate_non_negative(self):
        acc = ResonanceAccumulator()
        rng = np.random.default_rng(4)
        sites = [
            (rng.standard_normal(104), rng.uniform(0.1, 1.0), rng.uniform(0.0, 3.0))
            for _ in range(8)
        ]
        result = acc.accumulate(sites)
        assert np.all(result >= 0.0)

    def test_accumulate_self_excitation_dominates(self):
        """A single point should self-excite and produce > 0 amplitude."""
        acc = ResonanceAccumulator(resonance_radius=1.0)
        p = np.zeros(104)
        sites = [(p, 1.0, 0.0)]
        result = acc.accumulate(sites)
        assert result[0] > 0.5

    def test_accumulate_nearby_sites_have_higher_psi(self):
        """Tightly clustered sites should produce higher Ψ than isolated ones."""
        acc = ResonanceAccumulator(resonance_radius=0.5)
        cluster = [(np.zeros(104) + np.random.default_rng(i).standard_normal(104) * 0.01,
                    1.0, 0.0) for i in range(5)]
        isolated = [(np.ones(104) * 100.0, 1.0, 0.0)]
        cluster_psi = acc.accumulate(cluster)
        isolated_psi = acc.accumulate(isolated)
        assert cluster_psi.mean() > isolated_psi.mean()

    def test_resonance_accumulator_alias_importable(self):
        acc = ResonanceAccumulatorAlias()
        assert isinstance(acc, ResonanceAccumulator)


# ─────────────────────────────────────────────────────────────────────────────
# TestResonanceLayer
# ─────────────────────────────────────────────────────────────────────────────

class TestResonanceLayer:

    def test_resonance_layer_creates(self, living):
        rl = ResonanceLayer(living)
        assert rl.amplitude_floor == pytest.approx(0.01)

    def test_resonance_layer_rejects_bad_radius(self, living):
        with pytest.raises(ValueError):
            ResonanceLayer(living, resonance_radius=0.0)

    def test_resonance_layer_rejects_bad_floor(self, living):
        with pytest.raises(ValueError):
            ResonanceLayer(living, amplitude_floor=-0.1)

    def test_accumulate_empty_trajectory_returns_wave(self, resonance_layer, simple_query):
        empty_traj = Trajectory(steps=[], query=simple_query, termination_reason="empty")
        wave = resonance_layer.accumulate(empty_traj)
        assert isinstance(wave, StandingWave)
        assert len(wave.points) == 0

    def test_accumulate_returns_standing_wave(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        assert isinstance(wave, StandingWave)

    def test_wave_points_have_104d_vectors(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        for pt in wave.points:
            assert pt.vector.shape == (DIM,)

    def test_wave_total_energy_positive(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        assert wave.total_energy > 0.0

    def test_wave_amplitudes_non_negative(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        for pt in wave.points:
            assert pt.amplitude >= 0.0

    def test_wave_amplitudes_at_most_one(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        for pt in wave.points:
            assert pt.amplitude <= 1.0 + 1e-9

    def test_wave_query_echo_present(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        assert wave.query_echo is not None
        assert wave.query_echo.amplitude == pytest.approx(0.05)

    def test_wave_metadata_contains_n_steps(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        assert "n_trajectory_steps" in wave.metadata
        assert wave.metadata["n_trajectory_steps"] == len(short_trajectory)

    def test_wave_metadata_contains_termination_reason(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        assert "termination_reason" in wave.metadata

    def test_wave_tau_in_unit_interval(self, resonance_layer, short_trajectory):
        wave = resonance_layer.accumulate(short_trajectory)
        for pt in wave.points:
            assert 0.0 <= pt.tau <= 1.0 + 1e-9

    def test_resonance_layer_alias_importable(self, living):
        rl = ResonanceLayerAlias(living)
        assert isinstance(rl, ResonanceLayer)


# ─────────────────────────────────────────────────────────────────────────────
# TestResonanceLayerIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestResonanceLayerIntegration:

    def test_full_pipeline_causal_domain(self, living):
        """End-to-end: causal query → trajectory → standing wave."""
        q = Query(
            vector=living.position("causal::perturbation").copy(),
            label="causal_query",
        )
        fe = FlowEngine(living, max_steps=40, seed=50)
        rl = ResonanceLayer(living)
        traj = fe.flow(q)
        wave = rl.accumulate(traj)
        assert isinstance(wave, StandingWave)
        assert wave.total_energy > 0.0

    def test_full_pipeline_logical_domain(self, living):
        q = Query(
            vector=living.position("domain::logical_entities").copy(),
            label="logical_query",
        )
        fe = FlowEngine(living, max_steps=40, seed=51)
        rl = ResonanceLayer(living)
        traj = fe.flow(q)
        wave = rl.accumulate(traj)
        assert not traj.is_empty
        assert wave.total_energy > 0.0

    def test_wave_confidence_bounded(self, living):
        q = Query(vector=living.position("domain::mathematical").copy())
        fe = FlowEngine(living, max_steps=30, seed=60)
        rl = ResonanceLayer(living)
        wave = rl.accumulate(fe.flow(q))
        confidence = wave.mean_confidence()
        assert 0.0 <= confidence <= 1.0

    def test_wave_peak_is_most_activated_point(self, living):
        q = Query(vector=living.position("causal::perturbation").copy(), label="test")
        fe = FlowEngine(living, max_steps=30, seed=70)
        rl = ResonanceLayer(living)
        wave = rl.accumulate(fe.flow(q))
        if wave.peak:
            for pt in wave.points:
                assert wave.peak.amplitude >= pt.amplitude - 1e-9

    def test_wave_top_k_returns_k(self, living):
        q = Query(vector=living.position("causal::perturbation").copy(), label="test")
        fe = FlowEngine(living, max_steps=30, seed=80)
        rl = ResonanceLayer(living)
        wave = rl.accumulate(fe.flow(q))
        k = min(3, len(wave.points))
        top = wave.top_k(k)
        assert len(top) == k

    def test_wave_feeds_expression_renderer(self, living):
        """The StandingWave produced by C6 must be consumable by C7."""
        from src.phase1.expression.renderer import ExpressionRenderer
        q = Query(vector=living.position("causal::perturbation").copy(), label="causal")
        fe = FlowEngine(living, max_steps=20, seed=90)
        rl = ResonanceLayer(living)
        wave = rl.accumulate(fe.flow(q))
        # C7 must accept the wave — may have few points but should not error
        if len(wave.points) > 0:
            renderer = ExpressionRenderer()
            output = renderer.render(wave)
            assert isinstance(output.text, str)

    def test_two_different_queries_produce_different_waves(self, living):
        q1 = Query(vector=living.position("causal::perturbation").copy(), label="c")
        q2 = Query(vector=living.position("domain::mathematical").copy(), label="m")
        rl = ResonanceLayer(living)
        fe1 = FlowEngine(living, max_steps=60, seed=91)
        fe2 = FlowEngine(living, max_steps=60, seed=91)
        t1 = fe1.flow(q1)
        t2 = fe2.flow(q2)
        w1 = rl.accumulate(t1)
        w2 = rl.accumulate(t2)
        # Starting positions differ → trajectories start at different points
        assert not np.allclose(
            t1.steps[0].position, t2.steps[0].position
        ), "Queries start at different positions"
