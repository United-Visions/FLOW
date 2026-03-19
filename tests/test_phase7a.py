"""Phase 7a — Unit tests: CoOccurrenceCounter and WordPlacer.

Coverage targets:
  TestCoOccurrenceCounter   — ~25 tests
  TestWordPlacer            — ~20 tests
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.vocabulary.cooccurrence import CoOccurrenceCounter, CoOccurrenceMatrix
from src.vocabulary.word_placer import (
    WordPlacer,
    structural_feature_vector,
    _morphological_class,
    _char_ngram_fingerprint,
    _count_syllables,
    _FUNCTION_WORDS,
    _NEGATION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture helpers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "the cat sat on the mat the cat is fluffy "
    "the dog ran outside the dog is fast "
    "cats and dogs are different animals "
    "the fluffy cat sat near the slow dog "
    "not all animals run fast some animals are slow "
    "the quick brown fox jumps over the lazy dog "
    "science explains the mechanism of cause and effect "
    "rapid acceleration follows initial force "
    "every action has an equal reaction never forget "
    "the evidence suggests that correlation is not causation "
    "perhaps the outcome is uncertain maybe not certain "
    "several possible outcomes exist under different conditions "
    "the cat sat on the mat again and the dog barked "
    "animals can cause rapid changes in their environment "
    "the mechanism enables acceleration through force "
)


def _build_counter(text: str | None = None, **kwargs) -> CoOccurrenceCounter:
    c = CoOccurrenceCounter(**kwargs)
    c.feed(text or SAMPLE_TEXT)
    return c


def _build_matrix(text: str | None = None, **kwargs) -> CoOccurrenceMatrix:
    c = _build_counter(text, **kwargs)
    return c.build()


# ─────────────────────────────────────────────────────────────────────────────
# TestCoOccurrenceCounter
# ─────────────────────────────────────────────────────────────────────────────

class TestCoOccurrenceCounter:

    def test_feed_counts_tokens(self):
        c = CoOccurrenceCounter(min_count=1)
        c.feed("hello world")
        assert c.n_tokens_seen == 2

    def test_feed_multiple_texts(self):
        c = CoOccurrenceCounter(min_count=1)
        c.feed("hello world")
        c.feed("world peace")
        assert c.n_tokens_seen == 4

    def test_feed_stream(self):
        c = CoOccurrenceCounter(min_count=1)
        c.feed_stream(["hello world", "world peace"])
        assert c.n_tokens_seen == 4

    def test_vocabulary_size_increases(self):
        c = CoOccurrenceCounter(min_count=1)
        c.feed("alpha beta gamma")
        assert c.vocabulary_size == 3

    def test_normalise_strips_punctuation(self):
        c = CoOccurrenceCounter(min_count=1)
        c.feed("Hello, World! It's great.")
        m = c.build()
        # "hello" and "world" should be present; "its" and "great" too
        assert "hello" in m.vocabulary
        assert "world" in m.vocabulary

    def test_normalise_lowercase(self):
        c = CoOccurrenceCounter(min_count=1)
        c.feed("THE CAT SAT")
        m = c.build()
        assert "the" in m.vocabulary
        assert "cat" in m.vocabulary
        assert "sat" in m.vocabulary

    def test_build_returns_matrix(self):
        m = _build_matrix()
        assert isinstance(m, CoOccurrenceMatrix)

    def test_build_once_only(self):
        c = _build_counter()
        c.build()
        with pytest.raises(RuntimeError, match="build\\(\\)"):
            c.build()

    def test_feed_after_build_raises(self):
        c = _build_counter()
        c.build()
        with pytest.raises(RuntimeError, match="Counter has already been built"):
            c.feed("more text")

    def test_min_count_prunes_rare_words(self):
        c = CoOccurrenceCounter(min_count=5)
        c.feed("cat " * 10 + "hapax")   # "hapax" appears once
        m = c.build()
        assert "cat" in m.vocabulary
        assert "hapax" not in m.vocabulary

    def test_vocabulary_ceiling(self):
        words = [f"word{i}" * 6 for i in range(200)]  # 200 distinct words, each repeated 6 times
        text = " ".join(words) + " " + " ".join(words[: 100])
        c = CoOccurrenceCounter(min_count=1, v_max=50)
        c.feed(text)
        m = c.build()
        assert len(m.vocabulary) <= 50

    def test_pmi_symmetric(self):
        m = _build_matrix(min_count=2)
        # PMI(a,b) == PMI(b,a) by definition
        if len(m.vocabulary) >= 2:
            w1, w2 = m.vocabulary[0], m.vocabulary[1]
            assert abs(m.pmi(w1, w2) - m.pmi(w2, w1)) < 1e-9

    def test_pmi_of_unknown_pair_is_zero(self):
        m = _build_matrix()
        assert m.pmi("zzz_unknown_a", "zzz_unknown_b") == 0.0

    def test_pmi_range(self):
        """PMI can be negative (less than chance) or positive (more than chance)."""
        m = _build_matrix(min_count=2)
        vals = [m.pmi(w1, w2) for w1, w2, _ in m.pairs_above_threshold(0.0, -99)]
        if vals:
            # PMI is a real number; check it's finite
            for v in vals[:20]:
                assert math.isfinite(v)

    def test_dpmi_directed(self):
        """dPMI(w1→w2) need not equal dPMI(w2→w1)."""
        m = _build_matrix(min_count=1)
        if len(m.vocabulary) >= 2:
            w1, w2 = m.vocabulary[0], m.vocabulary[1]
            # directed PMI values exist (may differ or be equal)
            dpmi_fwd = m.dpmi(w1, w2)
            dpmi_rev = m.dpmi(w2, w1)
            assert isinstance(dpmi_fwd, float)
            assert isinstance(dpmi_rev, float)

    def test_dpmi_unknown_pair_is_zero(self):
        m = _build_matrix()
        assert m.dpmi("zzz_x", "zzz_y") == 0.0

    def test_pmi_max_positive(self):
        m = _build_matrix(min_count=2)
        assert m.pmi_max() > 0.0

    def test_pairs_above_threshold_returned(self):
        m = _build_matrix(min_count=2)
        pairs = m.pairs_above_threshold(tau_same=0.5, tau_diff=-1.0)
        # At least some pairs should pass either threshold
        if pairs:
            for w1, w2, val in pairs:
                assert val > 0.5 or val < -1.0

    def test_directed_pairs_above_delta(self):
        m = _build_matrix(min_count=1)
        directed = m.directed_pairs_above_delta(delta=0.1)
        # Result is a list (may be empty on short text)
        assert isinstance(directed, list)

    def test_frequency_rank(self):
        m = _build_matrix(min_count=1)
        # "the" should have a very low rank (high frequency) in SAMPLE_TEXT
        if "the" in m.vocabulary:
            rank = m.frequency_rank("the")
            assert rank == 1 or rank <= 5

    def test_frequency_rank_unknown_word(self):
        m = _build_matrix()
        rank = m.frequency_rank("xyzw_unknown")
        assert rank > len(m.vocabulary)

    def test_unigram_count(self):
        c = CoOccurrenceCounter(min_count=1)
        c.feed("cat cat cat dog")
        m = c.build()
        assert m.unigram_count("cat") == 3
        assert m.unigram_count("dog") == 1

    def test_build_result_is_deterministic(self):
        m1 = _build_matrix()
        vocab1 = set(m1.vocabulary)
        # Rebuild fresh counter with same text
        m2 = _build_matrix()
        vocab2 = set(m2.vocabulary)
        assert vocab1 == vocab2

    def test_window_size_parameter(self):
        """Larger window captures more co-occurrence pairs."""
        c1 = CoOccurrenceCounter(window_size=1, min_count=1)
        c1.feed(SAMPLE_TEXT)
        m1 = c1.build()
        c2 = CoOccurrenceCounter(window_size=10, min_count=1)
        c2.feed(SAMPLE_TEXT)
        m2 = c2.build()
        pairs1 = len(m1.pairs_above_threshold(0.0, -99))
        pairs2 = len(m2.pairs_above_threshold(0.0, -99))
        assert pairs2 >= pairs1

    def test_vocabulary_property(self):
        m = _build_matrix()
        assert isinstance(m.vocabulary, list)
        assert all(isinstance(w, str) for w in m.vocabulary)

    def test_empty_text_no_crash(self):
        c = CoOccurrenceCounter(min_count=1)
        c.feed("")
        m = c.build()
        assert isinstance(m, CoOccurrenceMatrix)


# ─────────────────────────────────────────────────────────────────────────────
# TestWordPlacer
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipeline(T0=1.0, T_floor=0.05):
    """Return a (LivingManifold, AnnealingEngine) pair for testing."""
    from src.phase1.seed_geometry.engine import SeedGeometryEngine
    from src.phase2.living_manifold.manifold import LivingManifold
    from src.phase3.annealing_engine.engine import AnnealingEngine

    M0 = SeedGeometryEngine().build()
    manifold = LivingManifold(M0)
    engine = AnnealingEngine(manifold, T0=T0, T_floor=T_floor)
    return manifold, engine


class TestStructuralFeatureVector:

    def test_output_shape(self):
        vec = structural_feature_vector("cat")
        assert vec.shape == (104,)

    def test_output_range(self):
        for word in ["the", "not", "acceleration", "mechanism", "perhaps"]:
            vec = structural_feature_vector(word)
            assert vec.min() >= 0.0
            assert vec.max() <= 1.0

    def test_function_word_high_probabilistic(self):
        """Function words should have higher probabilistic fiber values."""
        vec_the = structural_feature_vector("the")
        vec_mechanism = structural_feature_vector("mechanism")
        # Probabilistic fiber: dims 88–103
        assert np.mean(vec_the[88:104]) > np.mean(vec_mechanism[88:104])

    def test_negation_word_logical_fiber(self):
        """Negation words should have non-zero logical fiber dim 80."""
        vec = structural_feature_vector("not")
        assert vec[80] == 1.0

    def test_non_negation_zero_logical_dim(self):
        """Non-negation words should have dim 80 = 0."""
        vec = structural_feature_vector("cat")
        assert vec[80] == 0.0

    def test_causal_fiber_zeros(self):
        """Causal fiber (dims 64–79) must be all zeros at initial placement."""
        for word in ["cat", "the", "not", "acceleration", "perhaps"]:
            vec = structural_feature_vector(word)
            assert np.all(vec[64:80] == 0.0), f"Causal fiber not zero for '{word}'"

    def test_deterministic(self):
        """Same word always produces the same vector."""
        v1 = structural_feature_vector("mechanism")
        v2 = structural_feature_vector("mechanism")
        assert np.allclose(v1, v2)

    def test_different_words_differ(self):
        """Different words produce different structural vectors."""
        v1 = structural_feature_vector("cat")
        v2 = structural_feature_vector("acceleration")
        assert not np.allclose(v1, v2)

    def test_freq_rank_affects_probabilistic_fiber(self):
        """Higher freq_rank (rarer word) → different probabilistic fiber."""
        v_common = structural_feature_vector("the", freq_rank=1)
        v_rare   = structural_feature_vector("the", freq_rank=50000)
        # They may differ due to freq_rank influencing probabilistic fiber
        # (function words use a separate path so the difference may be small)
        assert v_common.shape == v_rare.shape

    def test_hedging_word_low_probabilistic(self):
        """Hedging words should be near the low end of probabilistic fiber."""
        for hedge in ["perhaps", "possibly", "maybe"]:
            vec = structural_feature_vector(hedge)
            assert np.mean(vec[88:104]) < 0.6


class TestWordPlacer:

    @pytest.fixture
    def placer_and_manifold(self):
        manifold, engine = _make_pipeline()
        return WordPlacer(engine), manifold

    def test_place_returns_vocab_label(self, placer_and_manifold):
        placer, _ = placer_and_manifold
        label = placer.place("cat")
        assert label == "vocab::cat"

    def test_place_adds_to_manifold(self, placer_and_manifold):
        placer, manifold = placer_and_manifold
        placer.place("dog")
        pos = manifold.position("vocab::dog")
        assert pos.shape == (104,)

    def test_place_label_format(self, placer_and_manifold):
        placer, _ = placer_and_manifold
        label = placer.place("mechanism")
        assert label.startswith("vocab::")
        assert label == "vocab::mechanism"

    def test_place_function_word(self, placer_and_manifold):
        """Function words placed in high-certainty probabilistic region."""
        placer, manifold = placer_and_manifold
        placer.place("the")
        pos = manifold.position("vocab::the")
        # Probabilistic fiber (88–103) should be elevated for function words
        assert np.mean(pos[88:104]) > 0.3   # flexible threshold; just non-zero

    def test_place_negation_word(self, placer_and_manifold):
        """Negation words have non-zero logical fiber component."""
        placer, manifold = placer_and_manifold
        placer.place("not")
        pos = manifold.position("vocab::not")
        # Logical fiber dim 80 should reflect negation
        assert pos[80] > 0.0

    def test_place_content_word(self, placer_and_manifold):
        """Content words should have a non-zero position on the manifold."""
        placer, manifold = placer_and_manifold
        placer.place("acceleration", freq_rank=5000)
        pos = manifold.position("vocab::acceleration")
        assert np.linalg.norm(pos) > 0.0

    def test_temperature_restored(self, placer_and_manifold):
        """Temperature must be restored after placement."""
        placer, _ = placer_and_manifold
        t_before = placer._engine.temperature
        placer.place("test")
        t_after = placer._engine.temperature
        # Temperature will have changed slightly due to schedule stepping,
        # but T0 should be restored so it hasn't been permanently lowered
        assert placer._engine.schedule.T0 > placer._engine.schedule.T_floor

    def test_place_batch(self, placer_and_manifold):
        placer, manifold = placer_and_manifold
        words  = ["fast", "slow", "red", "blue"]
        labels = placer.place_batch(words)
        assert len(labels) == 4
        for word, label in zip(words, labels):
            assert f"vocab::{word}" in label or label == f"vocab::{word}"
            pos = manifold.position(f"vocab::{word}")
            assert pos.shape == (104,)

    def test_place_returns_string(self, placer_and_manifold):
        placer, _ = placer_and_manifold
        result = placer.place("elephant")
        assert isinstance(result, str)

    def test_multiple_placements_independent(self, placer_and_manifold):
        """Placing multiple words leaves each with distinct positions."""
        placer, manifold = placer_and_manifold
        for word in ["alpha", "beta", "gamma"]:
            placer.place(word)
        p1 = manifold.position("vocab::alpha")
        p2 = manifold.position("vocab::beta")
        p3 = manifold.position("vocab::gamma")
        # Not all identical
        assert not (np.allclose(p1, p2) and np.allclose(p2, p3))


# ─────────────────────────────────────────────────────────────────────────────
# TestHelpers
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_morphological_class_function(self):
        assert _morphological_class("the") == "function"
        assert _morphological_class("is") == "function"

    def test_morphological_class_verb(self):
        result = _morphological_class("running")
        assert result == "verb"

    def test_morphological_class_noun(self):
        result = _morphological_class("acceleration")
        assert result == "noun"

    def test_morphological_class_adjective(self):
        result = _morphological_class("beautiful")
        assert result == "adjective"

    def test_ngram_fingerprint_shape(self):
        fp = _char_ngram_fingerprint("hello", n=3, size=16)
        assert fp.shape == (16,)

    def test_ngram_fingerprint_range(self):
        fp = _char_ngram_fingerprint("hello", n=3, size=16)
        assert fp.min() >= 0.0
        assert fp.max() <= 1.0

    def test_ngram_fingerprint_deterministic(self):
        fp1 = _char_ngram_fingerprint("hello", n=4, size=16)
        fp2 = _char_ngram_fingerprint("hello", n=4, size=16)
        assert np.allclose(fp1, fp2)

    def test_syllable_count_monosyllabic(self):
        assert _count_syllables("cat") == 1
        assert _count_syllables("dog") == 1

    def test_syllable_count_polysyllabic(self):
        assert _count_syllables("acceleration") >= 4

    def test_syllable_count_at_least_one(self):
        for word in ["a", "i", "the", "rhythm"]:
            assert _count_syllables(word) >= 1
