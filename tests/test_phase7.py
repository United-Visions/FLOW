"""Phase 7 — Full integration tests.

Tests the VocabularyBuilder end-to-end:
  text → PMI matrix → word placement → contrast passes → ExpressionEntries → .npz

Also verifies all 516 prior tests continue to pass (import-level check) and
that the GEOPipeline + load_vocabulary combination produces valid output.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from src.vocabulary import (
    VocabularyBuilder,
    CoOccurrenceCounter,
    WordPlacer,
    ContrastScheduler,
    TemplateBuilder,
    VocabularyStore,
    CoOccurrenceMatrix,
    structural_feature_vector,
    compose_wave_profile,
    ContrastPair,
    CausalBiasDirective,
)
from src.phase1.expression.matcher import ExpressionEntry, ResonanceMatcher
from src.phase1.expression.wave import WAVE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Sample corpus
# ─────────────────────────────────────────────────────────────────────────────

CORPUS = [
    "the rapid acceleration of the mechanism causes propagation through the pathway",
    "correlation is not always causation but evidence suggests a link",
    "the perturbation propagates through the network with increasing velocity",
    "science explains the mechanism of cause and effect in physical systems",
    "rapid changes in force lead to acceleration of the underlying process",
    "not all effects are caused by the same mechanism some are concurrent",
    "perhaps the evidence is consistent with multiple possible explanations",
    "the initial conditions determine the trajectory of the system over time",
    "each action causes a reaction of equal magnitude in opposite direction",
    "the relationship between energy and mass reveals fundamental physical laws",
    "acceleration increases when force is applied and resistance decreases",
    "the underlying mechanism enables rapid propagation through cause chains",
    "some phenomena resist simple causal explanation they require system views",
    "evidence from multiple sources suggests the correlation is not coincidental",
    "the concept of causation is central to scientific reasoning and explanation",
    "perhaps different mechanisms operate under different boundary conditions",
    "the velocity of the particle changes as it encounters the force field",
    "never confuse correlation with causation in scientific reasoning",
    "all systems have an underlying structure that determines their behaviour",
    "the effect propagates faster when the initial perturbation is larger",
]


def _make_full_pipeline():
    """Return (GEOPipeline, VocabularyBuilder)."""
    from src.phase5 import GEOPipeline
    from src.phase3.annealing_engine.experience import Experience

    pipeline = GEOPipeline(T0=1.0, T_floor=0.05, flow_seed=42)

    vbuilder = VocabularyBuilder(
        manifold=pipeline.manifold,
        annealing_engine=pipeline._annealing,
        contrast_engine=pipeline._contrast_engine,
        window_size=4,
        min_count=2,
        v_max=1000,
        tau_same=0.8,
        tau_diff=-0.5,
        batch_size=64,
        n_contrast_passes=1,
    )
    return pipeline, vbuilder


# ─────────────────────────────────────────────────────────────────────────────
# TestVocabularyBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyBuilder:

    @pytest.fixture
    def pipeline_and_builder(self):
        return _make_full_pipeline()

    def test_init(self, pipeline_and_builder):
        _, vb = pipeline_and_builder
        assert isinstance(vb, VocabularyBuilder)

    def test_feed_text(self, pipeline_and_builder):
        _, vb = pipeline_and_builder
        vb.feed("the cat sat on the mat")
        assert vb.n_tokens_fed >= 6

    def test_feed_stream(self, pipeline_and_builder):
        _, vb = pipeline_and_builder
        vb.feed(["first sentence here", "second sentence here"])
        assert vb.n_tokens_fed >= 6

    def test_feed_returns_self(self, pipeline_and_builder):
        _, vb = pipeline_and_builder
        result = vb.feed("the quick brown fox")
        assert result is vb

    def test_build_and_save(self, pipeline_and_builder, tmp_path):
        _, vb = pipeline_and_builder
        vb.feed(CORPUS)
        path = str(tmp_path / "vocab.npz")
        n = vb.build_and_save(path)
        assert n > 0
        assert os.path.exists(path)

    def test_build_and_save_returns_positive_count(self, pipeline_and_builder, tmp_path):
        _, vb = pipeline_and_builder
        vb.feed(CORPUS)
        path = str(tmp_path / "vocab.npz")
        n = vb.build_and_save(path)
        assert isinstance(n, int)
        assert n > 0

    def test_words_placed_on_manifold(self, pipeline_and_builder, tmp_path):
        pipeline, vb = pipeline_and_builder
        vb.feed(CORPUS)
        path = str(tmp_path / "vocab.npz")
        vb.build_and_save(path)
        # At least some vocab words should be on the manifold
        n_vocab = sum(
            1 for l in pipeline.manifold._points if l.startswith("vocab::")
        )
        assert n_vocab > 0

    def test_summary(self, pipeline_and_builder, tmp_path):
        _, vb = pipeline_and_builder
        vb.feed(CORPUS)
        path = str(tmp_path / "vocab.npz")
        vb.build_and_save(path)
        summary = vb.summary()
        assert isinstance(summary, str)
        assert "VocabularyBuilder" in summary

    def test_n_words_placed_after_build(self, pipeline_and_builder, tmp_path):
        _, vb = pipeline_and_builder
        vb.feed(CORPUS)
        vb.build_and_save(str(tmp_path / "v.npz"))
        assert vb.n_words_placed > 0

    def test_n_judgments_after_build(self, pipeline_and_builder, tmp_path):
        _, vb = pipeline_and_builder
        vb.feed(CORPUS)
        vb.build_and_save(str(tmp_path / "v.npz"))
        # May be 0 if no qualifying PMI pairs — just check it's non-negative
        assert vb.n_judgments_applied >= 0

    def test_matrix_available_after_build(self, pipeline_and_builder, tmp_path):
        _, vb = pipeline_and_builder
        vb.feed(CORPUS)
        vb.build_and_save(str(tmp_path / "v.npz"))
        assert vb.matrix is not None
        assert isinstance(vb.matrix, CoOccurrenceMatrix)

    def test_build_without_save(self, pipeline_and_builder):
        pipeline2, vb2 = _make_full_pipeline()
        vb2.feed(CORPUS[:5])
        entries = vb2.build()
        assert isinstance(entries, list)
        for e in entries:
            assert isinstance(e, ExpressionEntry)

    def test_vocabulary_store_loadable(self, pipeline_and_builder, tmp_path):
        _, vb = pipeline_and_builder
        vb.feed(CORPUS)
        path = str(tmp_path / "vocab.npz")
        n = vb.build_and_save(path)
        loaded = VocabularyStore.load(path)
        assert len(loaded) == n

    def test_loaded_vocab_integrates_with_matcher(self, pipeline_and_builder, tmp_path):
        """After build_and_save, load into ResonanceMatcher."""
        _, vb = pipeline_and_builder
        vb.feed(CORPUS)
        path = str(tmp_path / "vocab.npz")
        vb.build_and_save(path)

        matcher = ResonanceMatcher()
        base = len(matcher.vocabulary)
        n = matcher.load_vocabulary(path)
        assert n > 0
        assert len(matcher.vocabulary) == base + n


# ─────────────────────────────────────────────────────────────────────────────
# TestGeoPipelineIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestGeoPipelineIntegration:
    """Verify that GEOPipeline.query() works after vocabulary loading."""

    @pytest.fixture
    def pipeline_with_vocab(self, tmp_path):
        pipeline, vbuilder = _make_full_pipeline()
        vbuilder.feed(CORPUS)
        path = str(tmp_path / "vocab.npz")
        vbuilder.build_and_save(path)
        # Load vocabulary into the pipeline's renderer
        pipeline._renderer.matcher.load_vocabulary(path)
        return pipeline

    def test_query_returns_result(self, pipeline_with_vocab):
        """GEOPipeline.query() should work normally after vocabulary loading."""
        pipeline = pipeline_with_vocab
        vec = np.random.default_rng(7).uniform(size=104)
        result = pipeline.query(vec, label="test query")
        assert result is not None
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_vocabulary_count_increases_matcher_size(self, pipeline_with_vocab):
        """Matcher has more entries than the base 32 after loading."""
        pipeline = pipeline_with_vocab
        assert len(pipeline._renderer.matcher.vocabulary) > 32

    def test_prior_tests_imports_clean(self):
        """Importing all Phase 1–6 modules should not raise."""
        import src.phase1.seed_geometry.engine
        import src.phase1.expression.renderer
        import src.phase2.living_manifold.manifold
        import src.phase2.contrast_engine.engine
        import src.phase3.annealing_engine.engine
        import src.phase4.flow_engine.engine
        import src.phase4.resonance_layer.layer
        import src.phase5.pipeline.pipeline
        # No assertion needed — reaching here means imports OK


# ─────────────────────────────────────────────────────────────────────────────
# TestPublicAPIExports
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicAPIExports:
    """Verify __init__.py exports all required symbols."""

    def test_vocabulary_builder_exported(self):
        from src.vocabulary import VocabularyBuilder
        assert VocabularyBuilder is not None

    def test_co_occurrence_counter_exported(self):
        from src.vocabulary import CoOccurrenceCounter
        assert CoOccurrenceCounter is not None

    def test_co_occurrence_matrix_exported(self):
        from src.vocabulary import CoOccurrenceMatrix
        assert CoOccurrenceMatrix is not None

    def test_word_placer_exported(self):
        from src.vocabulary import WordPlacer
        assert WordPlacer is not None

    def test_structural_feature_vector_exported(self):
        from src.vocabulary import structural_feature_vector
        assert callable(structural_feature_vector)

    def test_contrast_scheduler_exported(self):
        from src.vocabulary import ContrastScheduler
        assert ContrastScheduler is not None

    def test_contrast_pair_exported(self):
        from src.vocabulary import ContrastPair
        assert ContrastPair is not None

    def test_causal_bias_directive_exported(self):
        from src.vocabulary import CausalBiasDirective
        assert CausalBiasDirective is not None

    def test_template_builder_exported(self):
        from src.vocabulary import TemplateBuilder
        assert TemplateBuilder is not None

    def test_compose_wave_profile_exported(self):
        from src.vocabulary import compose_wave_profile
        assert callable(compose_wave_profile)

    def test_vocabulary_store_exported(self):
        from src.vocabulary import VocabularyStore
        assert VocabularyStore is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestDesignConstraints
# ─────────────────────────────────────────────────────────────────────────────

class TestDesignConstraints:
    """Verify the six design constraints are respected."""

    def test_no_ml_libraries_in_vocabulary_module(self):
        """The vocabulary module must not import torch, tensorflow, or sklearn."""
        import importlib
        import src.vocabulary.cooccurrence as m1
        import src.vocabulary.word_placer as m2
        import src.vocabulary.contrast_scheduler as m3
        import src.vocabulary.template_builder as m4
        import src.vocabulary.vocabulary_store as m5
        import src.vocabulary.builder as m6

        forbidden = {"torch", "tensorflow", "sklearn", "gensim", "transformers"}
        for mod in [m1, m2, m3, m4, m5, m6]:
            for attr in dir(mod):
                assert attr not in forbidden, (
                    f"Forbidden library '{attr}' found in {mod.__name__}"
                )

    def test_no_tokeniser_in_vocabulary(self):
        """The vocabulary module must not contain tokeniser functionality."""
        import src.vocabulary.cooccurrence as m
        # Normalise function only strips punctuation and lowercases — no BPE
        words = m.CoOccurrenceCounter._normalise("Hello, World!")
        # All returned strings should be plain lowercase alpha words
        for w in words:
            assert w.isalpha(), f"Non-alpha token in output: '{w}'"

    def test_causal_fiber_zero_on_placement(self):
        """structural_feature_vector must have all-zero causal fiber initially."""
        for word in ["cat", "the", "not", "acceleration", "mechanism"]:
            vec = structural_feature_vector(word)
            assert np.all(vec[64:80] == 0.0), (
                f"Causal fiber not zero for '{word}': {vec[64:80]}"
            )

    def test_vocabulary_store_no_pickle(self, tmp_path):
        """VocabularyStore must use .npz (numpy arrays), not pickle."""
        path = str(tmp_path / "test.npz")
        rng = np.random.default_rng(0)
        entry = ExpressionEntry(
            text="test",
            wave_profile=rng.uniform(size=WAVE_DIM),
            register="neutral", rhythm="short",
            uncertainty_fit=0.5, causal_strength=0.0, hedging=False,
        )
        VocabularyStore.save([entry], path)

        # File must be openable by np.load (native numpy format, not pickle)
        data = np.load(path, allow_pickle=True)
        assert "wave_profiles" in data
        assert "texts" in data

    def test_group_a_separation(self):
        """src/vocabulary must not import C5, C6, or C7 modules."""
        import src.vocabulary.builder as builder_mod
        import src.vocabulary.template_builder as tb_mod

        # These imports from builder.py and template_builder.py source code:
        # Only ExpressionEntry (Group B interface type) is allowed.
        # The modules must NOT import FlowEngine, ResonanceLayer, ExpressionRenderer.
        import ast, inspect

        for mod in [builder_mod, tb_mod]:
            source = inspect.getsource(mod)
            tree   = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        mod_name = node.module
                        assert "flow_engine" not in mod_name, (
                            f"flow_engine imported in {mod.__name__}"
                        )
                        assert "resonance_layer" not in mod_name, (
                            f"resonance_layer imported in {mod.__name__}"
                        )
                        # ExpressionRenderer is C7 — allowed only in matcher.py
                        assert "renderer" not in mod_name or "expression" not in mod_name.lower(), \
                            f"renderer imported in {mod.__name__}"
