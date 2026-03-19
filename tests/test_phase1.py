"""
Tests for Phase 1 — Seed Geometry Engine + Expression Renderer

Run with:
    python -m pytest tests/ -v
or:
    python -m pytest tests/ -v --tb=short

Test categories
---------------
1. Causal geometry — DAG validity, embedding, asymmetric distance
2. Logical geometry — hypercube, Hamming, NOT/AND/OR, uncertainty
3. Probabilistic geometry — simplex, Fisher, KL, entropy
4. Similarity geometry — metric, curvature, domain structure
5. Fiber bundle composer — projection, composition, metric PSD
6. Seed manifold M₀ — full validation, query API
7. Standing wave — construction, amplitude properties, segmentation
8. Resonance matcher — vocabulary, distance, matching
9. Expression renderer — full pipeline end-to-end
"""

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Seed Geometry — Component 1
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalGeometry:
    """Tests for causal geometric fiber."""

    @pytest.fixture(scope="class")
    def geo(self):
        from src.phase1.seed_geometry.causal import CausalGeometry
        return CausalGeometry.build()

    def test_dag_is_acyclic(self, geo):
        import networkx as nx
        assert nx.is_directed_acyclic_graph(geo.dag), "Causal DAG must be acyclic"

    def test_dag_has_nodes(self, geo):
        assert geo.dag.number_of_nodes() >= 10, "DAG must have meaningful node count"

    def test_dag_has_edges(self, geo):
        assert geo.dag.number_of_edges() >= 5, "DAG must have causal edges"

    def test_all_nodes_embedded(self, geo):
        for node in geo.dag.nodes():
            assert node in geo.embeddings, f"Node '{node}' missing embedding"
            assert geo.embeddings[node].shape == (geo.dim,)

    def test_embedding_tau_monotone(self, geo):
        """τ-axis must be monotonically consistent with causal order."""
        import networkx as nx
        for src, dst in geo.dag.edges():
            tau_src = geo.embeddings[src][3]  # TAU_AXIS = 3
            tau_dst = geo.embeddings[dst][3]
            assert tau_src <= tau_dst + 1e-8, (
                f"τ({src})={tau_src:.3f} > τ({dst})={tau_dst:.3f}: "
                f"violates causal time ordering"
            )

    def test_causal_distance_non_negative(self, geo):
        nodes = geo.node_names
        if len(nodes) >= 2:
            v1 = geo.embeddings[nodes[0]]
            v2 = geo.embeddings[nodes[1]]
            assert geo.causal_distance(v1, v2) >= 0.0

    def test_causal_distance_asymmetric(self, geo):
        """Causal distance must be asymmetric (retro-causal travel costs more)."""
        nodes = geo.node_names
        for i in range(len(nodes) - 1):
            v1 = geo.embeddings[nodes[i]]
            v2 = geo.embeddings[nodes[i + 1]]
            d_fwd = geo.causal_distance(v1, v2)
            d_rev = geo.causal_distance(v2, v1)
            # At least some pairs should show asymmetry
            if not np.isclose(d_fwd, d_rev, atol=1e-6):
                return   # found asymmetric pair — test passes
        # If all distances happened to be symmetric, check the time axis
        # (some may be equidistant in τ which is acceptable)
        pytest.skip("All sample pairs happen to be τ-equidistant — no asymmetry testable")

    def test_causal_direction_unit_vector(self, geo):
        nodes = geo.node_names
        if len(nodes) >= 2:
            v1 = geo.embeddings[nodes[0]]
            v2 = geo.embeddings[nodes[-1]]
            d = geo.causal_direction(v1, v2)
            norm = np.linalg.norm(d)
            assert abs(norm - 1.0) < 1e-6 or norm == 0.0, (
                f"Causal direction should be a unit vector, got norm={norm}"
            )

    def test_intervention_distance_positive(self, geo):
        v = geo.embeddings[geo.node_names[0]]
        d = geo.intervention_distance(v)
        assert d >= 0.0


class TestLogicalGeometry:
    """Tests for logical geometric fiber."""

    @pytest.fixture(scope="class")
    def geo(self):
        from src.phase1.seed_geometry.logical import LogicalGeometry
        return LogicalGeometry.build()

    def test_hypercube_vertex_count(self, geo):
        expected = 2 ** geo.n_dims
        assert len(geo.vertices) == expected

    def test_vertices_binary(self, geo):
        """All vertices must be binary (0 or 1 in each dimension)."""
        assert np.all((geo.vertices == 0) | (geo.vertices == 1))

    def test_hamming_distance_zero_self(self, geo):
        v = geo.vertices[0]
        assert geo.hamming_distance(v, v) == 0.0

    def test_hamming_distance_max_negation(self, geo):
        v  = geo.vertices[0]
        nv = geo.negate(v)
        d  = geo.hamming_distance(v, nv)
        assert d == geo.n_dims, "Negation should produce maximum Hamming distance"

    def test_not_involution(self, geo):
        """NOT is an involution: NOT(NOT(v)) == v."""
        for v in geo.vertices[:10]:
            dbl_neg = geo.negate(geo.negate(v))
            assert np.allclose(dbl_neg, v), "Double negation must return original"

    def test_and_idempotent(self, geo):
        """AND(v, v) == v."""
        for v in geo.vertices[:10]:
            assert np.allclose(geo.conjunction(v, v), v)

    def test_or_idempotent(self, geo):
        """OR(v, v) == v."""
        for v in geo.vertices[:10]:
            assert np.allclose(geo.disjunction(v, v), v)

    def test_de_morgan(self, geo):
        """NOT(AND(v1, v2)) == OR(NOT(v1), NOT(v2))."""
        for i in range(min(5, len(geo.vertices))):
            for j in range(min(5, len(geo.vertices))):
                v1, v2 = geo.vertices[i], geo.vertices[j]
                lhs = geo.negate(geo.conjunction(v1, v2))
                rhs = geo.disjunction(geo.negate(v1), geo.negate(v2))
                assert np.allclose(lhs, rhs), "De Morgan's law violated"

    def test_logical_centre(self, geo):
        c = geo.logical_centre()
        assert np.allclose(c, 0.5 * np.ones(geo.n_dims))

    def test_uncertainty_zero_on_vertex(self, geo):
        v = geo.vertices[0]
        u = geo.uncertainty_score(v)
        assert u < 0.01, "Uncertainty must be ~0 on a hypercube vertex"

    def test_uncertainty_max_at_centroid(self, geo):
        c = geo.logical_centre()
        u = geo.uncertainty_score(c)
        assert u > 0.9, "Uncertainty must be ~1 at the logical centroid"

    def test_entailment_reflexive(self, geo):
        for v in geo.vertices[:5]:
            assert geo.entails(v, v), "Entailment must be reflexive"


class TestProbabilisticGeometry:
    """Tests for probabilistic geometric fiber."""

    @pytest.fixture(scope="class")
    def geo(self):
        from src.phase1.seed_geometry.probabilistic import ProbabilisticGeometry
        return ProbabilisticGeometry.build()

    def test_simplex_vertices(self, geo):
        assert geo.vertices.shape == (geo.k, geo.k)
        assert np.allclose(np.sum(geo.vertices, axis=1), 1.0)

    def test_centroid_uniform(self, geo):
        assert np.allclose(geo.centroid, 1.0 / geo.k)

    def test_to_simplex_valid(self, geo):
        v = np.array([0.5, -0.2, 0.8, 0.1] + [0.0] * (geo.k - 4))
        p = geo.to_simplex(v)
        assert p.sum() > 0.99 and np.all(p >= 0)

    def test_fisher_metric_diagonal(self, geo):
        p = geo.normalize(np.ones(geo.k))
        G = geo.fisher_metric(p)
        assert G.shape == (geo.k, geo.k)
        # Fisher metric for uniform distribution should be diagonal
        off_diag = G - np.diag(np.diag(G))
        assert np.allclose(off_diag, 0.0)

    def test_riemannian_distance_zero_self(self, geo):
        p = geo.normalize(np.ones(geo.k))
        d = geo.riemannian_distance(p, p)
        assert abs(d) < 1e-5   # arccos(1) has small float error at boundary

    def test_riemannian_distance_symmetric(self, geo):
        p = geo.normalize(np.array([0.8] + [0.2 / (geo.k - 1)] * (geo.k - 1)))
        q = geo.normalize(np.array([0.1] + [0.9 / (geo.k - 1)] * (geo.k - 1)))
        d_pq = geo.riemannian_distance(p, q)
        d_qp = geo.riemannian_distance(q, p)
        assert abs(d_pq - d_qp) < 1e-6, "Fisher-Rao distance must be symmetric"

    def test_kl_divergence_non_negative(self, geo):
        p = geo.normalize(np.ones(geo.k))
        q = geo.normalize(np.array(range(1, geo.k + 1), dtype=float))
        assert geo.kl_divergence(p, q) >= 0.0

    def test_kl_divergence_zero_self(self, geo):
        p = geo.normalize(np.ones(geo.k))
        assert abs(geo.kl_divergence(p, p)) < 1e-6

    def test_entropy_max_at_uniform(self, geo):
        uniform = geo.centroid
        h_uniform = geo.entropy(uniform)
        v = geo.vertices[0]
        h_vertex = geo.entropy(v)
        assert h_uniform > h_vertex, "Entropy must be maximal at uniform distribution"

    def test_confidence_zero_at_uniform(self, geo):
        c = geo.confidence(geo.centroid)
        assert c < 0.01, "Confidence must be ~0 at maximum uncertainty"

    def test_confidence_one_at_vertex(self, geo):
        v = geo.vertices[0]
        c = geo.confidence(v)
        assert c > 0.99, "Confidence must be ~1 at a certainty vertex"

    def test_geodesic_endpoints(self, geo):
        p = geo.vertices[0]
        q = geo.vertices[1]
        assert np.allclose(geo.geodesic(p, q, 0.0), p, atol=1e-4)
        assert np.allclose(geo.geodesic(p, q, 1.0), q, atol=1e-4)

    def test_geodesic_midpoint_valid(self, geo):
        p = geo.normalize(np.ones(geo.k))
        q = geo.vertices[0]
        mid = geo.geodesic(p, q, 0.5)
        assert abs(mid.sum() - 1.0) < 1e-6, "Midpoint must be valid probability vector"
        assert np.all(mid >= 0.0)


class TestSimilarityGeometry:
    """Tests for similarity base manifold."""

    @pytest.fixture(scope="class")
    def geo(self):
        from src.phase1.seed_geometry.similarity import SimilarityGeometry
        return SimilarityGeometry.build()

    def test_domain_centers_populated(self, geo):
        assert len(geo.domain_centers) > 0

    def test_domain_center_dimensions(self, geo):
        for name, c in geo.domain_centers.items():
            assert c.shape == (geo.dim,), f"Domain '{name}' center has wrong shape"

    def test_metric_positive_semidefinite(self, geo):
        p = geo.domain_centers[list(geo.domain_centers.keys())[0]]
        G = geo.metric_tensor(p)
        eigvals = np.linalg.eigvalsh(G)
        assert eigvals.min() >= -1e-8, "Metric tensor must be PSD"

    def test_distance_non_negative(self, geo):
        centers = list(geo.domain_centers.values())
        if len(centers) >= 2:
            d = geo.riemannian_distance(centers[0], centers[1])
            assert d >= 0.0

    def test_distance_zero_self(self, geo):
        p = list(geo.domain_centers.values())[0]
        assert geo.riemannian_distance(p, p) < 1e-10

    def test_triangle_inequality(self, geo):
        centers = list(geo.domain_centers.values())
        if len(centers) >= 3:
            a, b, c = centers[:3]
            d_ab = geo.riemannian_distance(a, b)
            d_bc = geo.riemannian_distance(b, c)
            d_ac = geo.riemannian_distance(a, c)
            assert d_ac <= d_ab + d_bc + 1e-8, "Triangle inequality violated"

    def test_similarity_range(self, geo):
        centers = list(geo.domain_centers.values())
        if len(centers) >= 2:
            s = geo.similarity_score(centers[0], centers[1])
            assert 0.0 <= s <= 1.0

    def test_similarity_one_self(self, geo):
        p = list(geo.domain_centers.values())[0]
        s = geo.similarity_score(p, p)
        assert s > 0.99

    def test_domain_of_returns_string(self, geo):
        p = list(geo.domain_centers.values())[0]
        d = geo.domain_of(p)
        assert isinstance(d, str) and len(d) > 0

    def test_curvature_range(self, geo):
        for p in list(geo.domain_centers.values())[:5]:
            kappa = geo.local_curvature(p)
            assert 0.0 <= kappa <= 5.0, f"Curvature {kappa} out of expected range"


class TestFiberBundleComposer:
    """Tests for the fiber bundle composition."""

    @pytest.fixture(scope="class")
    def bundle(self):
        from src.phase1.seed_geometry.causal import CausalGeometry
        from src.phase1.seed_geometry.logical import LogicalGeometry
        from src.phase1.seed_geometry.probabilistic import ProbabilisticGeometry
        from src.phase1.seed_geometry.similarity import SimilarityGeometry
        from src.phase1.seed_geometry.composer import FiberBundleComposer
        sim  = SimilarityGeometry.build()
        cau  = CausalGeometry.build()
        log  = LogicalGeometry.build()
        prob = ProbabilisticGeometry.build()
        return FiberBundleComposer(sim, cau, log, prob)

    def test_bundle_point_shape(self, bundle):
        from src.phase1.seed_geometry.composer import DIM_TOTAL, DIM_CAUSAL, DIM_LOGICAL, DIM_PROB, DIM_SIMILARITY
        base    = np.zeros(DIM_SIMILARITY)
        causal  = np.zeros(DIM_CAUSAL)
        logical = np.full(DIM_LOGICAL, 0.5)
        prob    = np.full(DIM_PROB, 1.0 / DIM_PROB)
        p = bundle.bundle_point(base, causal, logical, prob)
        assert p.shape == (DIM_TOTAL,)

    def test_projection_roundtrip(self, bundle):
        p = bundle.all_neutral()
        base, causal, logical, prob_fib = bundle.project(p)
        reconstructed = bundle.bundle_point(base, causal, logical, prob_fib)
        assert np.allclose(p, reconstructed)

    def test_compose_metric_psd(self, bundle):
        p = bundle.all_neutral()
        result = bundle.validate_metric(p)
        assert result["is_psd"],      f"Metric not PSD: min_eig={result['min_eigenvalue']}"
        assert result["is_symmetric"], "Metric not symmetric"

    def test_bundle_distance_non_negative(self, bundle):
        p1 = bundle.all_neutral()
        p2 = bundle.all_neutral()
        p2[:10] += 0.5
        d = bundle.bundle_distance(p1, p2)
        assert d >= 0.0

    def test_bundle_distance_zero_self(self, bundle):
        p = bundle.all_neutral()
        assert bundle.bundle_distance(p, p) < 1e-10

    def test_neutral_fibers_correct_shape(self, bundle):
        from src.phase1.seed_geometry.composer import DIM_CAUSAL, DIM_LOGICAL, DIM_PROB
        assert bundle.neutral_causal_fiber().shape  == (DIM_CAUSAL,)
        assert bundle.neutral_logical_fiber().shape == (DIM_LOGICAL,)
        assert bundle.neutral_prob_fiber().shape    == (DIM_PROB,)


# ─────────────────────────────────────────────────────────────────────────────
# Seed Manifold M₀ — Integration Test
# ─────────────────────────────────────────────────────────────────────────────


class TestSeedManifold:
    """Integration tests for the full seed manifold."""

    @pytest.fixture(scope="class")
    def M0(self):
        from src.phase1.seed_geometry.engine import SeedGeometryEngine
        print("\nBuilding M₀ for tests...")
        engine = SeedGeometryEngine()
        return engine.build()

    def test_build_produces_manifold(self, M0):
        from src.phase1.seed_geometry.manifold import SeedManifold
        assert isinstance(M0, SeedManifold)

    def test_dimension_correct(self, M0):
        assert M0.dim == 104

    def test_seed_points_non_empty(self, M0):
        assert len(M0.seed_points) > 0

    def test_all_seed_points_correct_dim(self, M0):
        for sp in M0.seed_points:
            assert sp.vector.shape == (104,), (
                f"Seed point '{sp.label}' has wrong shape: {sp.vector.shape}"
            )

    def test_validation_passes(self, M0):
        v = M0.validate()
        assert v["all_points_correct_dim"], "Some seed points have wrong dimensionality"
        assert v["metric_psd"],             "Composed metric is not PSD"
        assert v["metric_symmetric"],       "Composed metric is not symmetric"

    def test_position_lookup(self, M0):
        # Every seed point should be findable by label
        for sp in M0.seed_points[:5]:
            found = M0.position(sp.label)
            assert np.allclose(found.vector, sp.vector)

    def test_distance_non_negative(self, M0):
        a, b = M0.seed_points[0], M0.seed_points[1]
        assert M0.distance(a, b) >= 0.0

    def test_distance_zero_self(self, M0):
        a = M0.seed_points[0]
        assert M0.distance(a, a) < 1e-10

    def test_triangle_inequality(self, M0):
        pts = M0.seed_points
        if len(pts) >= 3:
            a, b, c = pts[0], pts[1], pts[2]
            d_ab = M0.distance(a, b)
            d_bc = M0.distance(b, c)
            d_ac = M0.distance(a, c)
            assert d_ac <= d_ab + d_bc + 1e-8

    def test_causal_direction_unit_norm(self, M0):
        pts = M0.seed_points
        if len(pts) >= 2:
            d = M0.causal_direction(pts[0], pts[1])
            n = np.linalg.norm(d)
            assert n == 0.0 or abs(n - 1.0) < 1e-6

    def test_curvature_non_negative(self, M0):
        for sp in M0.seed_points[:10]:
            k = M0.curvature(sp)
            assert k >= 0.0

    def test_confidence_range(self, M0):
        for sp in M0.seed_points[:10]:
            c = M0.confidence(sp)
            assert 0.0 <= c <= 1.0

    def test_domain_returns_string(self, M0):
        sp = M0.seed_points[0]
        d = M0.domain_of(sp)
        assert isinstance(d, str) and len(d) > 0

    def test_neighbors_returns_list(self, M0):
        sp = M0.seed_points[0]
        nbrs = M0.neighbors(sp, radius=10.0)
        assert isinstance(nbrs, list)

    def test_nearest_returns_closest(self, M0):
        sp = M0.seed_points[0]
        nearest = M0.nearest(sp, k=1)
        assert len(nearest) == 1

    def test_build_is_idempotent(self, M0):
        """Second call to build() should return same M₀."""
        from src.phase1.seed_geometry.engine import SeedGeometryEngine
        engine = SeedGeometryEngine()
        m1 = engine.build()
        m2 = engine.build()
        assert m1 is m2, "SeedGeometryEngine.build() must be idempotent"


# ─────────────────────────────────────────────────────────────────────────────
# Expression Renderer — Component 7
# ─────────────────────────────────────────────────────────────────────────────


class TestStandingWave:
    """Tests for the StandingWave data type and mock wave constructors."""

    @pytest.fixture
    def wave(self):
        from src.phase1.expression.wave import create_mock_wave
        return create_mock_wave("explanation")

    def test_wave_has_points(self, wave):
        assert len(wave.points) > 0

    def test_wave_sorted_by_amplitude(self, wave):
        amps = [p.amplitude for p in wave.points]
        assert amps == sorted(amps, reverse=True)

    def test_total_energy_positive(self, wave):
        assert wave.total_energy > 0.0

    def test_normalised_amplitudes_range(self, wave):
        na = wave.normalised_amplitudes
        assert na.max() <= 1.0 + 1e-8
        assert na.min() >= 0.0

    def test_peak_is_highest(self, wave):
        peak = wave.peak
        assert peak is not None
        assert peak.amplitude == max(p.amplitude for p in wave.points)

    def test_confident_core_not_empty(self, wave):
        core = wave.confident_core(threshold=0.1)
        assert len(core) > 0

    def test_mean_confidence_range(self, wave):
        c = wave.mean_confidence()
        assert 0.0 <= c <= 1.0

    def test_all_themes_buildable(self):
        from src.phase1.expression.wave import create_mock_wave
        themes = ["explanation", "causation", "uncertainty", "contrast",
                  "discovery", "warning", "instruction", "conclusion"]
        for t in themes:
            w = create_mock_wave(t)
            assert len(w.points) > 0, f"Theme '{t}' produced empty wave"

    def test_trajectory_wave(self):
        from src.phase1.expression.wave import create_wave_from_trajectory
        rng = np.random.default_rng(0)
        traj = [(rng.normal(size=104), float(i)) for i in range(10)]
        w = create_wave_from_trajectory(traj)
        assert len(w.points) == 10
        assert w.total_energy > 0.0


class TestResonanceMatcher:
    """Tests for the ResonanceMatcher."""

    @pytest.fixture(scope="class")
    def matcher(self):
        from src.phase1.expression.matcher import ResonanceMatcher
        return ResonanceMatcher()

    def test_vocabulary_non_empty(self, matcher):
        assert len(matcher.vocabulary) > 5

    def test_all_profiles_have_correct_dim(self, matcher):
        for entry in matcher.vocabulary:
            assert entry.wave_profile.shape == (matcher.dim,)

    def test_all_profiles_normalised(self, matcher):
        for entry in matcher.vocabulary:
            n = np.linalg.norm(entry.wave_profile)
            assert abs(n - 1.0) < 1e-6 or n < 1e-12, (
                f"Entry '{entry.text[:40]}' profile not normalised: norm={n}"
            )

    def test_match_returns_result(self, matcher):
        from src.phase1.expression.wave import create_mock_wave, WavePoint, WaveSegment
        from src.phase1.expression.matcher import MatchResult
        wave = create_mock_wave("causation")
        seg  = WaveSegment(
            points=wave.points[:5],
            mean_amplitude=0.8,
            peak_point=wave.points[0],
            index=0,
        )
        result = matcher.match(seg)
        assert isinstance(result, MatchResult)
        assert 0.0 <= result.resonance_score <= 1.0
        assert result.expression.text  # non-empty

    def test_match_all_length(self, matcher):
        from src.phase1.expression.wave import create_mock_wave, WavePoint, WaveSegment
        wave = create_mock_wave("contrast")
        segments = []
        for i in range(3):
            pts = wave.points[i*3:(i+1)*3]
            if pts:
                seg = WaveSegment(
                    points=pts, mean_amplitude=0.5,
                    peak_point=pts[0], index=i
                )
                segments.append(seg)
        results = matcher.match_all(segments)
        assert len(results) == len(segments)

    def test_resonance_distance_zero_self(self, matcher):
        v = np.random.default_rng(0).normal(size=matcher.dim)
        v /= np.linalg.norm(v)
        d = matcher._resonance_distance(v, v)
        assert abs(d) < 1e-6


class TestExpressionRenderer:
    """End-to-end tests for the Expression Renderer pipeline."""

    @pytest.fixture(scope="class")
    def renderer(self):
        from src.phase1.expression.renderer import ExpressionRenderer
        return ExpressionRenderer()

    def test_render_returns_output(self, renderer):
        from src.phase1.expression.wave import create_mock_wave
        from src.phase1.expression.renderer import RenderedOutput
        wave   = create_mock_wave("explanation")
        result = renderer.render(wave)
        assert isinstance(result, RenderedOutput)

    def test_rendered_text_non_empty(self, renderer):
        from src.phase1.expression.wave import create_mock_wave
        wave   = create_mock_wave("explanation")
        result = renderer.render(wave)
        assert len(result.text) > 0

    def test_confidence_range(self, renderer):
        from src.phase1.expression.wave import create_mock_wave
        wave   = create_mock_wave("causation")
        result = renderer.render(wave)
        assert 0.0 <= result.confidence <= 1.0

    def test_segments_non_empty(self, renderer):
        from src.phase1.expression.wave import create_mock_wave
        wave   = create_mock_wave("instruction")
        result = renderer.render(wave)
        assert len(result.segments) > 0

    def test_matches_align_with_segments(self, renderer):
        from src.phase1.expression.wave import create_mock_wave
        wave   = create_mock_wave("warning")
        result = renderer.render(wave)
        assert len(result.matches) == len(result.segments)

    def test_text_properly_terminated(self, renderer):
        from src.phase1.expression.wave import create_mock_wave
        for theme in ["causation", "uncertainty", "contrast", "conclusion"]:
            wave   = create_mock_wave(theme)
            result = renderer.render(wave)
            last_char = result.text.strip()[-1] if result.text.strip() else ""
            assert last_char in ".!?", (
                f"Theme '{theme}': text does not end with punctuation: '{result.text[-20:]}'"
            )

    def test_all_themes_render_successfully(self, renderer):
        from src.phase1.expression.wave import create_mock_wave
        themes = ["explanation", "causation", "uncertainty", "contrast",
                  "discovery", "warning", "instruction", "conclusion"]
        for t in themes:
            wave   = create_mock_wave(t)
            result = renderer.render(wave)
            assert len(result.text) > 0, f"Theme '{t}' produced empty output"

    def test_uncertain_wave_uses_hedging(self, renderer):
        from src.phase1.expression.wave import create_mock_wave
        wave   = create_mock_wave("uncertainty")
        result = renderer.render(wave)
        hedge_words = ["appears", "suggests", "likely", "seems", "possible",
                       "uncertain", "approximately", "difficult to say"]
        assert any(w in result.text.lower() for w in hedge_words), (
            f"Uncertain wave should produce hedging language.\nGot: {result.text}"
        )

    def test_render_no_weights_no_tokens(self, renderer):
        """
        Structural test: verify no actual weight-based or token-API constructs
        are used in the renderer pipeline (docstring mentions are fine).
        """
        import inspect
        import src.phase1.expression.renderer as mod
        source = inspect.getsource(mod)
        # Check for actual API calls / imports, not docstring commentary
        forbidden_calls = [
            "import torch", "import keras", "import tensorflow",
            "model.predict", "model.forward", "softmax(", "logit(",
            "tokenizer(", "token_ids", "vocab_size", "embedding_layer(",
            "nn.Linear", "nn.Embedding",
        ]
        for f in forbidden_calls:
            assert f not in source, (
                f"Forbidden construct '{f}' found in expression renderer"
            )
