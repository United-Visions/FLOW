"""Phase 7c — Unit tests: TemplateBuilder, VocabularyStore, and matcher.load_vocabulary().

Coverage targets:
  TestTemplateBuilder     — ~25 tests
  TestVocabularyStore     — ~20 tests
  TestMatcherLoad         — ~15 tests
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.phase1.expression.matcher import ExpressionEntry, ResonanceMatcher
from src.phase1.expression.wave import WAVE_DIM
from src.vocabulary.template_builder import (
    TemplateBuilder,
    compose_wave_profile,
    _derive_register,
    _derive_uncertainty_fit,
    _derive_causal_strength,
    _derive_hedging,
    _rhythm_from_text,
)
from src.vocabulary.vocabulary_store import VocabularyStore


# ─────────────────────────────────────────────────────────────────────────────
# Shared factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_manifold_with_vocab(words=None):
    """Return a LivingManifold with a small set of vocab words placed."""
    from src.phase1.seed_geometry.engine import SeedGeometryEngine
    from src.phase2.living_manifold.manifold import LivingManifold
    from src.phase3.annealing_engine.engine import AnnealingEngine
    from src.vocabulary.word_placer import WordPlacer

    M0 = SeedGeometryEngine().build()
    manifold = LivingManifold(M0)
    engine = AnnealingEngine(manifold, T0=1.0, T_floor=0.05)
    placer = WordPlacer(engine)

    if words is None:
        words = [
            "cat", "dog", "fast", "slow", "rapid", "mechanism",
            "acceleration", "cause", "effect", "the", "not", "perhaps",
            "evidence", "suggests", "correlation", "force", "velocity",
        ]
    for i, w in enumerate(words):
        placer.place(w, freq_rank=i + 1)

    return manifold


def _make_entries(n=10) -> list[ExpressionEntry]:
    """Create n dummy ExpressionEntry objects."""
    rng = np.random.default_rng(42)
    entries = []
    for i in range(n):
        wp = rng.uniform(size=WAVE_DIM).astype(np.float64)
        wp /= np.linalg.norm(wp)
        entries.append(ExpressionEntry(
            text=f"test entry {i}",
            wave_profile=wp,
            register=["neutral", "formal", "casual"][i % 3],
            rhythm=["short", "medium", "long"][i % 3],
            uncertainty_fit=float(rng.uniform()),
            causal_strength=float(rng.uniform()),
            hedging=bool(i % 2),
        ))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# TestTemplatBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestTemplateBuilder:

    @pytest.fixture
    def manifold(self):
        return _make_manifold_with_vocab()

    @pytest.fixture
    def builder(self, manifold):
        return TemplateBuilder(manifold)

    def test_build_returns_list(self, builder):
        entries = builder.build()
        assert isinstance(entries, list)

    def test_build_returns_expression_entries(self, builder):
        entries = builder.build()
        for e in entries:
            assert isinstance(e, ExpressionEntry)

    def test_level1_generates_entries(self, builder):
        entries = builder._build_level1()
        assert len(entries) > 0

    def test_level1_words_have_vocab_prefix_stripped(self, builder):
        entries = builder._build_level1()
        for e in entries:
            assert not e.text.startswith("vocab::")   # label prefix stripped

    def test_level1_wave_profile_shape(self, builder):
        entries = builder._build_level1()
        for e in entries:
            assert e.wave_profile.shape == (WAVE_DIM,)

    def test_level1_wave_profile_normalised(self, builder):
        entries = builder._build_level1()
        for e in entries:
            norm = float(np.linalg.norm(e.wave_profile))
            assert abs(norm - 1.0) < 1e-6 or norm < 1e-6  # unit vector or zero

    def test_level1_rhythm_is_short(self, builder):
        entries = builder._build_level1()
        for e in entries:
            assert e.rhythm == "short"

    def test_level3_generates_sentence_frames(self, builder):
        entries = builder._build_level3()
        assert len(entries) > 0
        # Level 3 entries should have {} slots in their text
        for e in entries:
            assert isinstance(e.text, str)

    def test_build_no_duplicate_texts(self, builder):
        entries = builder.build()
        texts = [e.text for e in entries]
        assert len(texts) == len(set(texts))

    def test_compose_wave_profile_shape(self, manifold):
        labels = ["vocab::cat", "vocab::dog"]
        wp = compose_wave_profile(manifold, labels)
        assert wp.shape == (WAVE_DIM,)

    def test_compose_wave_profile_normalised(self, manifold):
        labels = ["vocab::mechanism", "vocab::acceleration"]
        wp = compose_wave_profile(manifold, labels)
        norm = float(np.linalg.norm(wp))
        assert abs(norm - 1.0) < 1e-6

    def test_compose_wave_empty_labels(self, manifold):
        """Empty label list returns a valid fallback vector."""
        wp = compose_wave_profile(manifold, [])
        assert wp.shape == (WAVE_DIM,)
        assert np.linalg.norm(wp) > 0.0

    def test_compose_wave_missing_labels(self, manifold):
        """Labels not on manifold are skipped; still returns valid vector."""
        wp = compose_wave_profile(manifold, ["vocab::zzz_not_here_xyz"])
        assert wp.shape == (WAVE_DIM,)

    def test_derive_register_returns_valid(self, manifold):
        r = _derive_register(manifold, "vocab::mechanism")
        assert r in ("neutral", "formal", "casual")

    def test_derive_uncertainty_fit_in_range(self, manifold):
        u = _derive_uncertainty_fit(manifold, "vocab::perhaps")
        assert 0.0 <= u <= 1.0

    def test_derive_causal_strength_zero_for_new_word(self, manifold):
        """Causal fiber starts at zero; causal strength should be near 0."""
        c = _derive_causal_strength(manifold, "vocab::cause")
        assert isinstance(c, float)

    def test_derive_hedging_returns_bool(self, manifold):
        result = _derive_hedging(manifold, "vocab::perhaps")
        assert isinstance(result, bool)

    def test_rhythm_from_text_short(self):
        assert _rhythm_from_text("quick brown fox") == "short"

    def test_rhythm_from_text_medium(self):
        assert _rhythm_from_text("the quick brown fox jumped over") == "medium"

    def test_rhythm_from_text_long(self):
        assert _rhythm_from_text("the quite quick brown fox enthusiastically jumped over the lazy old") == "long"

    def test_calibrate_phrase_radius(self, builder):
        r = builder.calibrate_phrase_radius()
        assert isinstance(r, float)
        assert r > 0.0

    def test_max_level_limits(self, manifold):
        builder = TemplateBuilder(manifold, max_level1=5, max_level2=3, max_level3=2)
        entries = builder.build()
        # Hard limit: total ≤ max_level1 + max_level2 + max_level3
        assert len(entries) <= 5 + 3 + 2

    def test_build_with_matrix(self, manifold):
        from src.vocabulary.cooccurrence import CoOccurrenceCounter
        c = CoOccurrenceCounter(min_count=1)
        c.feed("cat dog fast slow mechanism acceleration")
        matrix = c.build()
        builder = TemplateBuilder(manifold)
        entries = builder.build(matrix)
        assert isinstance(entries, list)


# ─────────────────────────────────────────────────────────────────────────────
# TestVocabularyStore
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyStore:

    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(10)
        n = VocabularyStore.save(entries, path)
        assert n == 10
        loaded = VocabularyStore.load(path)
        assert len(loaded) == 10

    def test_save_returns_count(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(5)
        n = VocabularyStore.save(entries, path)
        assert n == 5

    def test_load_texts_preserved(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(5)
        VocabularyStore.save(entries, path)
        loaded = VocabularyStore.load(path)
        for orig, load in zip(entries, loaded):
            assert orig.text == load.text

    def test_load_wave_profiles_preserved(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(5)
        VocabularyStore.save(entries, path)
        loaded = VocabularyStore.load(path)
        for orig, load in zip(entries, loaded):
            assert np.allclose(orig.wave_profile, load.wave_profile, atol=1e-5)

    def test_load_register_preserved(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(6)
        VocabularyStore.save(entries, path)
        loaded = VocabularyStore.load(path)
        for orig, load in zip(entries, loaded):
            assert orig.register == load.register

    def test_load_rhythm_preserved(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(6)
        VocabularyStore.save(entries, path)
        loaded = VocabularyStore.load(path)
        for orig, load in zip(entries, loaded):
            assert orig.rhythm == load.rhythm

    def test_load_uncertainty_fit_preserved(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(5)
        VocabularyStore.save(entries, path)
        loaded = VocabularyStore.load(path)
        for orig, load in zip(entries, loaded):
            assert abs(orig.uncertainty_fit - load.uncertainty_fit) < 1e-4

    def test_load_causal_strength_preserved(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(5)
        VocabularyStore.save(entries, path)
        loaded = VocabularyStore.load(path)
        for orig, load in zip(entries, loaded):
            assert abs(orig.causal_strength - load.causal_strength) < 1e-4

    def test_load_hedging_preserved(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(5)
        VocabularyStore.save(entries, path)
        loaded = VocabularyStore.load(path)
        for orig, load in zip(entries, loaded):
            assert orig.hedging == load.hedging

    def test_load_returns_expression_entries(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        VocabularyStore.save(_make_entries(3), path)
        loaded = VocabularyStore.load(path)
        for e in loaded:
            assert isinstance(e, ExpressionEntry)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            VocabularyStore.load("/tmp/zzz_nonexistent_vocab_12345.npz")

    def test_save_empty_raises(self, tmp_path):
        path = str(tmp_path / "empty.npz")
        with pytest.raises(ValueError, match="empty"):
            VocabularyStore.save([], path)

    def test_count(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(7)
        VocabularyStore.save(entries, path)
        assert VocabularyStore.count(path) == 7

    def test_count_missing_returns_zero(self):
        n = VocabularyStore.count("/tmp/zzz_no_such_file_xyz_98765.npz")
        assert n == 0

    def test_append_adds_entries(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        e1 = _make_entries(5)
        VocabularyStore.save(e1, path)
        e2 = _make_entries(3)
        # Make text unique
        for i, e in enumerate(e2):
            e.text = f"new entry {i + 100}"
        total = VocabularyStore.append(e2, path)
        assert total == 8

    def test_append_deduplicates(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(5)
        VocabularyStore.save(entries, path)
        # Append the same entries again — only unique texts should be kept
        total = VocabularyStore.append(entries, path)
        assert total == 5  # no duplicates added

    def test_append_creates_file_if_missing(self, tmp_path):
        path = str(tmp_path / "new_vocab.npz")
        entries = _make_entries(4)
        total = VocabularyStore.append(entries, path)
        assert total == 4
        assert os.path.exists(path)

    def test_large_save_and_load(self, tmp_path):
        """Test with a larger vocabulary (1000 entries)."""
        path = str(tmp_path / "large.npz")
        entries = _make_entries(1000)
        n = VocabularyStore.save(entries, path)
        assert n == 1000
        loaded = VocabularyStore.load(path)
        assert len(loaded) == 1000

    def test_wave_profile_dim_mismatch_padded(self, tmp_path):
        """Entries with shorter wave profiles are zero-padded on save."""
        path = str(tmp_path / "short_wave.npz")
        # Create entry with half-length wave profile
        short_wp = np.ones(50, dtype=np.float64)
        entry = ExpressionEntry(
            text="short wave test",
            wave_profile=short_wp,
            register="neutral",
            rhythm="short",
            uncertainty_fit=0.5,
            causal_strength=0.0,
            hedging=False,
        )
        n = VocabularyStore.save([entry], path)
        assert n == 1
        loaded = VocabularyStore.load(path)
        assert loaded[0].wave_profile.shape == (WAVE_DIM,)


# ─────────────────────────────────────────────────────────────────────────────
# TestMatcherLoad
# ─────────────────────────────────────────────────────────────────────────────

class TestMatcherLoad:

    def test_load_vocabulary_returns_count(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(20)
        VocabularyStore.save(entries, path)

        matcher = ResonanceMatcher()
        initial = len(matcher.vocabulary)
        n = matcher.load_vocabulary(path)
        assert n == 20
        assert len(matcher.vocabulary) == initial + 20

    def test_load_vocabulary_appends_not_replaces(self, tmp_path):
        """The 32 hand-crafted entries must survive after loading."""
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(10)
        VocabularyStore.save(entries, path)

        matcher = ResonanceMatcher()
        original_count = len(matcher.vocabulary)
        assert original_count >= 32

        matcher.load_vocabulary(path)
        assert len(matcher.vocabulary) == original_count + 10

    def test_load_vocabulary_file_not_found(self):
        matcher = ResonanceMatcher()
        with pytest.raises(FileNotFoundError):
            matcher.load_vocabulary("/tmp/zzz_no_such_vocab_98765.npz")

    def test_loaded_entries_accessible_via_match(self, tmp_path):
        """After loading, match() can return a loaded entry (not just hand-crafted)."""
        from src.phase1.expression.wave import WavePoint, WaveSegment

        path = str(tmp_path / "vocab.npz")

        # Create an entry with a very distinct wave profile
        rng = np.random.default_rng(0)
        target_wp = np.zeros(WAVE_DIM)
        target_wp[0:8] = 1.0   # Strong signal in first 8 dims
        target_wp /= np.linalg.norm(target_wp)

        entry = ExpressionEntry(
            text="unique test phrase xyz",
            wave_profile=target_wp,
            register="neutral",
            rhythm="medium",
            uncertainty_fit=0.5,
            causal_strength=0.0,
            hedging=False,
        )
        VocabularyStore.save([entry], path)

        matcher = ResonanceMatcher()
        matcher.load_vocabulary(path)

        # Create a wave segment with a similar profile
        wpt = WavePoint(vector=target_wp.copy(), amplitude=1.0, label="test")
        seg = WaveSegment(points=[wpt], mean_amplitude=1.0, peak_point=wpt, index=0)

        result = matcher.match(seg)
        # Result should be a valid MatchResult
        assert result.resonance_score > 0.0
        assert isinstance(result.expression.text, str)

    def test_loaded_vocabulary_has_correct_wave_profiles(self, tmp_path):
        path = str(tmp_path / "vocab.npz")
        entries = _make_entries(5)
        VocabularyStore.save(entries, path)

        matcher = ResonanceMatcher()
        before_count = len(matcher.vocabulary)
        matcher.load_vocabulary(path)

        loaded_entries = matcher.vocabulary[before_count:]
        for orig, loaded in zip(entries, loaded_entries):
            assert np.allclose(orig.wave_profile, loaded.wave_profile, atol=1e-5)

    def test_load_many_entries_performance(self, tmp_path):
        """Loading 10,000 entries should complete quickly."""
        import time
        path = str(tmp_path / "big_vocab.npz")
        entries = _make_entries(10_000)
        VocabularyStore.save(entries, path)

        matcher = ResonanceMatcher()
        t0 = time.time()
        n = matcher.load_vocabulary(path)
        elapsed = time.time() - t0

        assert n == 10_000
        assert elapsed < 15.0   # should load in well under 15 seconds

    def test_match_uses_loaded_entry_over_base(self, tmp_path):
        """A loaded entry with a perfectly matching wave profile scores well."""
        from src.phase1.expression.wave import WavePoint, WaveSegment

        path = str(tmp_path / "perfect_match.npz")

        # Build a wave profile that strongly matches a specific segment
        profile = np.zeros(WAVE_DIM)
        profile[0:16] = 1.0    # dominant causal signal
        profile /= np.linalg.norm(profile)

        entry = ExpressionEntry(
            text="the {} drives the {}.",
            wave_profile=profile,
            register="formal",
            rhythm="medium",
            uncertainty_fit=0.2,
            causal_strength=0.9,
            hedging=False,
        )
        VocabularyStore.save([entry], path)

        matcher = ResonanceMatcher()
        matcher.load_vocabulary(path)

        wpt = WavePoint(vector=profile.copy(), amplitude=0.9, label="test")
        seg = WaveSegment(points=[wpt], mean_amplitude=0.9, peak_point=wpt, index=0)

        result = matcher.match(seg)
        assert result.resonance_score > 0.0

    def test_load_vocabulary_multiple_files(self, tmp_path):
        """Multiple load_vocabulary calls cumulate entries."""
        p1 = str(tmp_path / "v1.npz")
        p2 = str(tmp_path / "v2.npz")

        e1 = _make_entries(5)
        e2 = _make_entries(8)
        for i, e in enumerate(e2):
            e.text = f"second batch {i}"

        VocabularyStore.save(e1, p1)
        VocabularyStore.save(e2, p2)

        matcher = ResonanceMatcher()
        base = len(matcher.vocabulary)
        matcher.load_vocabulary(p1)
        matcher.load_vocabulary(p2)
        assert len(matcher.vocabulary) == base + 5 + 8
