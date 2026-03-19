# Phase 7 — Vocabulary Geometry Specification
## Extending ResonanceMatcher to ~100,000 Expression Entries via Manifold-Grounded Word Geometry

**Phase:** 7 (follows Phase 6 — Language Coherence, completed March 18 2026)  
**Version:** 0.1 — Foundational Spec  
**Status:** Pre-implementation — ready for AI agent pickup  
**Source of truth:** This document. Read it fully before writing any code.  
**Purpose:** Define the architecture, mathematics, and implementation plan for growing the C7 expression vocabulary from 32 hand-crafted templates to ~100,000 entries — without weights, tokens, training, or any external ML model.

---

## Context: Where We Are Coming From

Phases 1–6 are complete. The full C1–C7 pipeline is running end-to-end:

```
Experience → C3 AnnealingEngine → LivingManifold M(t)
                                       ↑
                              C4 ContrastEngine
                                       │
                              C5 FlowEngine → Trajectory
                                       │
                              C6 ResonanceLayer → StandingWave Ψ
                                       │
                              C7 ExpressionRenderer → text
```

**Phase 6 specifically fixed:** C6 now resolves WavePoint labels to real manifold concept names (`"mechanism"`, `"perturbation"`) instead of positional placeholders (`"flow_t0"`). C7 strips domain prefixes and uses template diversity penalties.

**The remaining gap** is that `{}` slots in C7 templates are still filled by manifold concept names — single words like `"mechanism"` or `"perturbation"`. These are real words, but the vocabulary is small (81 seed concepts) and the sentence frames are limited (32 hand-crafted templates). The output is coherent but narrow. Phase 7 fixes this by growing the vocabulary to ~100,000 geometrically-grounded entries.

**516/516 tests currently pass.** All must remain green after Phase 7.

---

## 0. The Problem

The current Expression Renderer (C7) has a **vocabulary gap**.

Each `{}` slot in a template is filled by a manifold label (`"mechanism"`, `"perturbation"`). This is a working pipeline, but it produces repetitive, narrow output because:

| Constraint | Current state | Target state |
|---|---|---|
| Template count | 32 hand-crafted entries | ~100,000 entries |
| Slot filling | Manifold label (concept name) | Rich English phrase |
| Wave profile source | Hand-tuned feature vectors | Derived from 104D manifold geometry |
| Coverage | ~6 grammatical patterns | Full repertoire of English expression |

The gap is **not architectural** — the pipeline (C5 → C6 → C7) is correct. The gap is a vocabulary problem: the ResonanceMatcher has too few entries, and each entry's wave profile is hand-tuned rather than grounded in the manifold.

**The core insight:** every English word or phrase can be treated as a concept that gets placed on the 104D manifold by the same C3/C4 machinery already used for all other knowledge. Once placed, its 104D position *is* its semantic wave profile. No new architecture is needed — only a systematic method for placing 100,000 linguistic concepts geometrically.

---

## 1. Design Constraint Compliance

All six constraints must be satisfied before any implementation begins.

| Constraint | Compliance |
|---|---|
| **NO WEIGHTS** | Co-occurrence counts are raw event statistics, not tunable parameters. SAME/DIFFERENT signals are derived by threshold on PMI — no gradient, no loss function. |
| **NO TOKENS** | Words are treated as *events* that produce geometric deformations, not as discrete symbol IDs. The co-occurrence window is a temporal locality window on a text stream, not a tokeniser. |
| **NO TRAINING** | Vocabulary growth runs as a continuous C3/C4 pass over text — the same operating mode as learning any other experience. No offline phase. |
| **LOCAL UPDATES** | Each word placement deforms nearby geometry only (C3 Gaussian falloff). The `deform_local` guarantee is unchanged. |
| **CAUSALITY FIRST** | PMI asymmetry (P(w₂\|w₁) ≠ P(w₁\|w₂)) is used to orient SAME/DIFFERENT judgments along the causal fiber (dims 64–79). Words that reliably *precede* others gain a positive causal direction component. |
| **SEPARATION** | Word geometry derivation is a Group A operation (manifold shaping). Template selection remains Group B (reasoning). No component crosses the boundary. |

---

## 2. Mathematical Foundations

### 2.1  Co-occurrence as Relational Signal

Given a text stream, a **co-occurrence event** is: word `w₁` appears within a context window of `w₂`.

Define the **Pointwise Mutual Information**:

$$\text{PMI}(w_1, w_2) = \log \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}$$

And the **asymmetric directed PMI** (for causality):

$$\text{dPMI}(w_1 \to w_2) = \log \frac{P(w_2 \mid w_1)}{P(w_2)}$$

These two quantities produce the contrast signal:

| Condition | Judgment | Manifold operation |
|---|---|---|
| PMI(w₁, w₂) > τ_same | **SAME** | Pull P(w₁), P(w₂) closer by α |
| PMI(w₁, w₂) < τ_diff | **DIFFERENT** | Push P(w₁), P(w₂) apart by β |
| τ_diff ≤ PMI ≤ τ_same | **NEUTRAL** | No operation (skip) |
| dPMI(w₁→w₂) ≫ dPMI(w₂→w₁) | **CAUSAL** | Bias displacement into causal fiber dims 64–79 |

Thresholds τ_same and τ_diff are not learned — they are set from first principles:
- τ_same = +1.0 (words that co-occur 2.7× more than chance are meaningfully related)
- τ_diff = −0.5 (words that co-occur less than chance are meaningfully distinct)

### 2.2  Initial Placement via C3 Annealing

Before C4 contrast can be applied, each word must have an initial position on M(t).

Initial placement vector for word `w` is a **structural feature vector** in 104D, derived entirely from surface properties — no corpus statistics yet:

**Similarity fiber (dims 0–63):**
- Character n-gram profile distributed across the 16-domain taxonomy embedding
- Morphological class (verb/noun/adjective/adverb/function) → coarse region assignment
- Syllable count and orthographic length → density/locality hint

**Causal fiber (dims 64–79):**
- All zeros initially — causal geometry emerges from directed PMI (§2.1)

**Logical fiber (dims 80–87):**
- Negation markers (`not`, `never`, `no`) → bit flip on dim 80
- Quantifiers (`all`, `some`, `none`) → graded bits on dims 81–83
- All other words → neutral Boolean state (0.5 · **1**)

**Probabilistic fiber (dims 88–103):**
- Function words (`the`, `a`, `is`) → high-certainty region (near `prob::certain_0`)
- Content words → moderate uncertainty (near `prob::maximal_uncertainty`)

This gives every word a geometrically meaningful starting position that is immediately consistent with the manifold's metric structure.

### 2.3  Iterative Contrast Refinement

After initial placement, words are refined by replaying co-occurrence judgments. This is the same C4 loop already in production:

```
for each (w₁, w₂, judgment) in contrast_stream:
    engine.judge(w₁, w₂, judgment, strength=pmi_to_strength(pmi))
```

The PMI value maps to judgment strength:

$$\text{strength}(w_1, w_2) = \min\left(1.0,\; \frac{|\text{PMI}(w_1,w_2)|}{\text{PMI}_{\text{max}}}\right)$$

This ensures high-PMI pairs produce strong geometric displacement, low-PMI pairs produce weak displacement — without any learned parameter controlling the mapping.

After K passes over the contrast stream, each word `w` has converged to a stable 104D position `P(w)` on M(t). This position **is** its semantic wave profile. No separate embedding step is needed.

### 2.4  Template Wave Profile Derivation

A template `T` with slot-text tokens `[w₁, w₂, ..., wₙ]` has a wave profile:

$$\Psi_T = \frac{\sum_{i=1}^{n} \omega_i \cdot P(w_i)}{\left\|\sum_{i=1}^{n} \omega_i \cdot P(w_i)\right\|}$$

Where the weight `ω_i` is determined by the word's **manifold density** ρ(P(wᵢ)):

$$\omega_i = 1 - \rho(P(w_i))$$

Low-density (novel) words contribute more to the template's meaning. High-density (common function words) contribute less. This is not a learned weight — it is a direct read from the manifold's density field.

The result is a 104D unit vector that encodes the phrase's meaning as a direction on the manifold. This vector slots directly into the `wave_profile` field of `ExpressionEntry` — the `ResonanceMatcher._resonance_distance()` function is unchanged.

### 2.5  Structural Metadata Derivation

Each `ExpressionEntry` also carries `register`, `rhythm`, `uncertainty_fit`, `causal_strength`, and `hedging`. These are derived geometrically:

| Field | Derivation |
|---|---|
| `register` | Mean probabilistic fiber norm: high → `"formal"`, low → `"casual"` |
| `rhythm` | Template word count: ≤4 → `"short"`, 5–9 → `"medium"`, ≥10 → `"long"` |
| `uncertainty_fit` | Mean amplitude at probabilistic fiber dims 88–103; normalised to [0,1] |
| `causal_strength` | Mean signed component on causal dims 64–79 |
| `hedging` | Presence in manifold neighbourhood of `"uncertainty"` / `"possibility"` seed concepts |

---

## 3. System Architecture

### 3.1  New Module: `src/vocabulary/`

```
src/vocabulary/
├── __init__.py               # exports VocabularyBuilder, VocabularyLoader
├── cooccurrence.py           # CoOccurrenceCounter — sliding-window PMI computation
├── word_placer.py            # WordPlacer — initial placement + C3 integration
├── contrast_scheduler.py     # ContrastScheduler — batched PMI → C4 judgment stream
├── template_builder.py       # TemplateBuilder — phrase → ExpressionEntry compiler
├── vocabulary_store.py       # VocabularyStore — on-disk serialisation (numpy .npz)
└── builder.py                # VocabularyBuilder — orchestrates all sub-components
```

This module sits in **Group A** (manifold shaping). It does not touch C5, C6, or C7 code paths. It only writes to M(t) and serialises the resulting vocabulary.

### 3.2  Modified: `src/phase1/expression/matcher.py`

One addition: `ResonanceMatcher` gains a `load_vocabulary(path)` method that reads a `VocabularyStore` into `self.vocabulary` without replacing the 32 hand-crafted entries (which remain as a bootstrapping floor).

```python
def load_vocabulary(self, path: str) -> int:
    """Load geometrically-derived entries from a VocabularyStore.
    Returns the number of entries loaded."""
```

Otherwise the matcher is unchanged. The 100K entries are just `ExpressionEntry` objects — same type, same matching logic, same distance function.

### 3.3  Data Flow

```
Text stream (any source)
        │
        ▼
  CoOccurrenceCounter
  ─────────────────────
  sliding window → raw co-occurrence counts
  counts → PMI matrix (symmetric) + dPMI matrix (directed)
        │
        ├─── new word w not on M(t)?
        │         │
        │         ▼
        │    WordPlacer
        │    ─────────────────────────────────
        │    structural feature vector → 104D
        │    Experience(vector, label=w) → C3.process()
        │    → word placed on M(t)
        │
        ▼
  ContrastScheduler
  ─────────────────────
  PMI(w₁,w₂) → judgment type + strength
  batches of 256 judgments → C4.judge()
  directed dPMI → causal fiber bias
        │
        ▼
  M(t) — all words now have stable 104D positions
        │
        ▼
  TemplateBuilder
  ─────────────────────
  enumerate phrase templates from word co-occurrence clusters
  compute Ψ_T for each template (§2.4)
  derive metadata fields (§2.5)
  → ExpressionEntry per template
        │
        ▼
  VocabularyStore
  ─────────────────────
  serialise entries to vocabulary.npz
        │
        ▼
  ResonanceMatcher.load_vocabulary("vocabulary.npz")
  → 100K entries available for resonance matching
```

---

## 4. Component Specifications

### 4.1  CoOccurrenceCounter

**Responsibility:** Convert a text stream into a PMI matrix and directed dPMI matrix.

**Input:** Any iterable of text strings (sentences, lines, paragraphs).  
**Output:** `CoOccurrenceMatrix` — symmetric `(V×V)` count array + row-wise `dPMI(w→*)`.

**Algorithm:**
```
for each sentence S:
    words = normalise(S)            # lowercase, strip punctuation, no stemming
    for each position i in words:
        for j in range(i+1, min(i+window_size+1, len(words))):
            w1, w2 = words[i], words[j]
            count[w1][w2] += 1
            count[w2][w1] += 1
            directed_count[w1→w2] += 1   # w1 precedes w2

pmi[w1][w2] = log(count[w1][w2] * N / (count[w1] * count[w2]))
dpmi[w1→w2] = log(directed_count[w1→w2] * N / (count[w1] * count[w2]))
```

**Parameters (not learned — derived from corpus statistics):**
- `window_size`: 5 words (standard linguistic co-occurrence window)
- `min_count`: 5 occurrences (prune words seen fewer than 5 times — geometric placement of hapax legomena is unreliable)
- `V_max`: 100,000 distinct words (vocabulary ceiling)

**What this is NOT:** This is not a tokeniser, not a neural embedding model, not a lookup table. Co-occurrence counts are raw geometric evidence — the same role that pixel co-occurrence plays in image geometry, or that temporal proximity plays in episodic memory. The counts are discarded after PMI is computed.

### 4.2  WordPlacer

**Responsibility:** Place each new vocabulary word onto M(t) via C3 Annealing.

**Input:** Word string + corpus frequency statistics.  
**Output:** Manifold label `"vocab::{word}"` placed at a 104D position.

**Algorithm:**
```
def place(word, freq_stats, annealing_engine):
    vec = structural_feature_vector(word, freq_stats)   # §2.2
    exp = Experience(
        vector=vec,
        label=f"vocab::{word}",
        source="vocabulary_geometry"
    )
    result = annealing_engine.process(exp)
    return result.placed_label
```

The `structural_feature_vector()` function (§2.2) is pure geometry — no corpus statistics except `freq_stats.rank` (frequency rank) which maps to a density hint in the probabilistic fiber.

**Constraint:** Words are placed with temperature `T = T_floor` (cold annealing). At cold temperature, placement is conservative — the word lands close to its structurally-derived initial position. Contrast refinement (§2.3) does the fine-grained positioning.

### 4.3  ContrastScheduler

**Responsibility:** Convert the PMI matrix into a stream of C4 judgments, batched for efficiency.

**Input:** `CoOccurrenceMatrix`.  
**Output:** Series of `ContrastPair` objects consumed by `ContrastEngine.judge()`.

**Algorithm:**
```
for each (w1, w2) pair with count > min_count:
    pmi = matrix.pmi(w1, w2)
    if pmi > τ_same:
        judgment = JudgmentType.SAME
    elif pmi < τ_diff:
        judgment = JudgmentType.DIFFERENT
    else:
        continue   # neutral — skip

    strength = min(1.0, abs(pmi) / pmi_max)
    yield ContrastPair(
        label_a=f"vocab::{w1}",
        label_b=f"vocab::{w2}",
        judgment=judgment,
        strength=strength
    )

# Directed causal judgments
for each (w1, w2) where dpmi[w1→w2] > dpmi[w2→w1] + δ_causal:
    yield CausalBiasDirective(
        label_a=f"vocab::{w1}",
        label_b=f"vocab::{w2}",
        direction=+1.0   # w1 tends to precede and cause w2
    )
```

**Batch size:** 256 judgments per call. After each batch, the manifold is allowed to settle (C3 density update) before the next batch runs. This ensures locality is maintained — a burst of 100K simultaneous judgments would violate the local-updates constraint.

### 4.4  TemplateBuilder

**Responsibility:** Construct `ExpressionEntry` objects from word positions on M(t).

**Input:** `LivingManifold` (fully populated with `vocab::*` points).  
**Output:** List of `ExpressionEntry` objects.

The builder generates templates at three levels of abstraction:

**Level 1 — Word entries (~50,000 entries)**  
Single words placed on the manifold. These fill `{}` slots with individual content words.

```python
ExpressionEntry(
    text="acceleration",
    wave_profile=manifold.position("vocab::acceleration")[:WAVE_DIM],
    register=derive_register(manifold, "vocab::acceleration"),
    rhythm="short",
    ...
)
```

**Level 2 — Phrase entries (~35,000 entries)**  
Noun phrases, verb phrases, and prepositional phrases built from co-clustering analysis of word positions. Words within geodesic radius `r < 0.3` on the similarity fiber are candidates for phrase combination.

```python
# e.g., "rapid acceleration" from "vocab::rapid" near "vocab::acceleration"
ExpressionEntry(
    text="rapid acceleration",
    wave_profile=compose_wave_profile(manifold, ["vocab::rapid", "vocab::acceleration"]),
    rhythm="medium",
    ...
)
```

**Level 3 — Sentence frame entries (~15,000 entries)**  
Complete sentence frames with `{}` slots derived from high-density co-occurrence clusters. These replace the current 32 hand-crafted templates and are generated by:
1. Find top-N dense clusters in M(t) across the similarity fiber
2. For each cluster, identify the grammatical skeleton (subject/verb/object pattern)
3. Generate a sentence frame where `{}` slots correspond to cluster centroids
4. Compute frame wave profile as weighted centroid of slot profiles (§2.4)

```python
ExpressionEntry(
    text="The {} enables {} through {}.",
    wave_profile=compose_wave_profile(manifold, ["vocab::enables", "vocab::through"]),
    causal_strength=0.8,
    rhythm="long",
    ...
)
```

**Total: ~100,000 entries** across all three levels, all with geometrically-derived wave profiles.

### 4.5  VocabularyStore

**Responsibility:** Serialise and deserialise vocabulary entries as pure numpy arrays (no pickle, no weights file format).

**Storage format:** `vocabulary.npz` containing:

```
wave_profiles   : float32 array (N × 104)   — one row per entry
register_ids    : uint8   array (N,)         — 0=neutral, 1=formal, 2=casual
rhythm_ids      : uint8   array (N,)         — 0=short, 1=medium, 2=long
uncertainty_fit : float32 array (N,)
causal_strength : float32 array (N,)
hedging_flags   : bool    array (N,)
texts           : object  array (N,)         — Python strings
```

Load into `ResonanceMatcher` in O(N) time with a single `np.load`. No model loading, no checkpoint restoration, no weight initialisation.

---

## 5. Implementation Plan

### Phase 7a — Corpus Infrastructure (est. 3–4 days)

| Step | File | Deliverable |
|---|---|---|
| 1 | `src/vocabulary/cooccurrence.py` | `CoOccurrenceCounter` class with PMI + dPMI computation |
| 2 | `src/vocabulary/word_placer.py` | `WordPlacer` class, structural feature vector function |
| 3 | `tests/test_phase7a.py` | Unit tests for co-occurrence math and placement |
| 4 | `tests/phase-7a_demo.py` | Demo: place 1,000 words from a sample text |

**Acceptance criterion:** 1,000 words placed on M(t) with geometrically sensible positions — function words cluster in high-certainty probabilistic fiber region, content words spread across similarity fiber, negation words are logical fiber neighbours.

### Phase 7b — Contrast Pass (est. 2–3 days)

| Step | File | Deliverable |
|---|---|---|
| 5 | `src/vocabulary/contrast_scheduler.py` | `ContrastScheduler` class, batched C4 judgment stream |
| 6 | Integration test: ContrastScheduler → ContrastEngine on 1,000-word vocab | All 516 prior tests still pass |
| 7 | `tests/test_phase7b.py` | Unit tests for PMI→judgment conversion, batch sizing |

**Acceptance criterion:** After K=3 passes of contrast judgments on 1,000 words — semantically related words (e.g., `"fast"` / `"rapid"`) have moved closer together on the similarity fiber; antonyms (`"increase"` / `"decrease"`) have moved further apart.

### Phase 7c — Template Construction (est. 3–4 days)

| Step | File | Deliverable |
|---|---|---|
| 8 | `src/vocabulary/template_builder.py` | `TemplateBuilder` class, all three levels (§4.4) |
| 9 | `src/vocabulary/vocabulary_store.py` | `VocabularyStore` serialise/load in .npz format |
| 10 | `src/phase1/expression/matcher.py` | Add `load_vocabulary(path)` method |
| 11 | `tests/test_phase7c.py` | Unit tests for template construction and store I/O |

**Acceptance criterion:** Load 10,000 entries into `ResonanceMatcher`. Query a wave segment — result uses a vocabulary entry derived from manifold geometry instead of a hand-crafted template.

### Phase 7d — Scale + Integration (est. 4–5 days)

| Step | File | Deliverable |
|---|---|---|
| 12 | `src/vocabulary/builder.py` | `VocabularyBuilder` orchestrator end-to-end |
| 13 | Full vocabulary build on a reference corpus (e.g., 10M word sample of public-domain text — no copyrighted material) | `vocabulary.npz` with ~100K entries |
| 14 | `tests/phase-7_demo.py` | Full pipeline demo: learn → contrast → query → language output showing vocabulary-derived phrases |
| 15 | `Phase-7_completed.md` | Completion record |

**Acceptance criterion:** `GEOPipeline.query()` output contains rich English phrases drawn from the 100K vocabulary, with template diversity ratio ≥ 0.8 across a 10-query evaluation suite.

---

## 6. Quality Metrics

These replace hand-inspection for evaluating whether the vocabulary is geometrically valid.

### 6.1  Neighbourhood Coherence

For a sample of 1,000 word pairs rated `"similar"` by human judgement (Wordsim-353 or equivalent):

$$\text{NbrCoherence} = \frac{1}{N} \sum_{(w_1,w_2) \in \text{similar}} \mathbf{1}\left[d_M(P(w_1), P(w_2)) < d_M(P(w_1), P(w_\text{rand}))\right]$$

Target: NbrCoherence > 0.75. This confirms geometry is semantically meaningful.

### 6.2  Causal Fiber Alignment

For a sample of known causal pairs (`"rain causes flooding"`, `"smoking causes cancer"`):

$$\text{CausalAlign} = \frac{1}{N} \sum_{(c,e) \in \text{causal\_pairs}} \mathbf{1}\left[\text{causal\_direction}(P(c), P(e)) > 0\right]$$

Target: CausalAlign > 0.65. This confirms dPMI is successfully shaping the causal fiber.

### 6.3  Template Diversity Ratio

For any 10 consecutive queries to `GEOPipeline.query()`:

$$\text{DiversityRatio} = \frac{\text{unique templates used}}{\text{total template slots filled}}$$

Target: DiversityRatio > 0.8. Currently 0.67 with 32 entries; 100K entries should easily exceed 0.8.

### 6.4  Vocabulary Coverage

Fraction of query trajectory WavePoints whose nearest vocabulary entry is within resonance distance < 0.3:

$$\text{Coverage} = \frac{|\{p \in \Psi : \min_E d(p, E) < 0.3\}|}{|\Psi|}$$

Target: Coverage > 0.90. Currently estimated < 0.40 with 32 entries.

---

## 7. Corpus Requirements and Dataset Selection

### 7.1  Hard Constraints

- **No copyrighted material.** Use only public-domain or openly-licensed text (CC-BY, CC-BY-SA, or public domain).
- **Raw text only.** Any HuggingFace dataset that exposes only `input_ids` (pre-tokenised) must be skipped — those are token-ID arrays, not text, and their use would violate NO TOKENS.
- **Size:** 10–50 million words is sufficient for PMI stability at 100K vocabulary size.
- **Diversity:** Mixed genres (narrative, expository, technical) to cover the full 104D manifold.

The corpus is consumed as a **stream** — it is never stored as a model input array and no sentence-level representation is retained. Only the `(w₁, w₂, count)` triples survive into the next phase. The `datasets` library from HuggingFace is permitted as a **data pipe only** — no models, no tokenisers, no embeddings are loaded from it.

```python
# Correct use — raw text stream, no model involvement
from datasets import load_dataset
ds = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)
for article in ds:
    raw_text = article["text"]    # plain Python string — identical to reading a .txt file
    co_occurrence_counter.feed(raw_text)
```

### 7.2  Recommended Datasets by Manifold Region

**Primary — Similarity Fiber (dims 0–63)**

| Dataset | HuggingFace ID | Licence | Words | Why |  
|---|---|---|---|---|
| Wikipedia | `wikimedia/wikipedia`, config `20220301.en` | CC-BY-SA | ~4B | Category hierarchy maps 1:1 onto the 16-domain taxonomy already seeded in dims 0–63. Single best general-purpose choice. |
| OpenWebText | `Skylion007/openwebtext` | Public domain curated | ~40B | Informal/conversational register. Wikipedia is entirely formal — without this, all words cluster in the `register: "formal"` region and C7 loses half its expressive range. |
| Project Gutenberg | `sedthh/gutenberg_english` | Public domain | ~3B | Narrative text. Provides temporal causation (`"she did X, then Y happened"`), emotional language, and long-range sentence frames essential for Level 3 template generation. |

**Causal Fiber — directed PMI signal (dims 64–79)**

| Dataset | HuggingFace ID | Licence | Words | Why |
|---|---|---|---|---|
| SciCite / S2ORC abstracts | `allenai/s2orc` | Open access | ~10B | Scientific abstracts are saturated with explicit causal language (`"X leads to"`, `"Y results in"`, `"Z was caused by"`). Dense directed PMI signal worth 10× their word count for causal fiber shaping. |
| BECAUSE corpus / causal pairs | Small structured datasets of cause-effect sentence pairs | Various | ~1M | Each sentence pair is a direct high-strength C4 CAUSAL judgment — far more efficient than inferring causality from raw co-occurrence. Feed as C4 inputs directly. |

**Logical Fiber — Boolean hypercube (dims 80–87)**

| Dataset | HuggingFace ID | Licence | Words | Why |
|---|---|---|---|---|
| SNLI | `stanfordnlp/snli` | CC-BY | ~570K pairs | Pairs labelled `entailment`/`contradiction`/`neutral`. Contradiction = DIFFERENT judgment. Entailment = SAME judgment. Already in exactly the format C4's `ContrastEngine.judge()` consumes. Sharpens the logical fiber immediately. |
| MultiNLI | `nyu-mll/multi_nli` | CC-BY | ~433K pairs | Same structure as SNLI but covers 10 text genres. Combined with SNLI, covers quantifiers (`all`, `some`, `none`) and negations (`not`, `never`) in controlled high-signal contexts. |

**Probabilistic Fiber — Fisher-Rao metric (dims 88–103)**

| Dataset | HuggingFace ID | Licence | Words | Why |
|---|---|---|---|---|
| Wikipedia (science articles) | Same as above | CC-BY-SA | — | Probability language (`"is likely"`, `"evidence suggests"`, `"consistent with"`) concentrated in science articles. Already present if Wikipedia is the primary corpus. |
| Epistemic hedging corpora | Small annotated datasets of hedging from academic writing | Various open | ~500K | Words like `"probably"`, `"certainly"`, `"possibly"`, `"might"` must land in the right Fisher-Rao region. Small but very high signal-to-noise for dims 88–103. |

### 7.3  Datasets to Avoid

| Dataset | Reason |
|---|---|
| C4 (`allenai/c4`) | Common Crawl source contains junk HTML, ad text, spam, near-duplicates. PMI computed over noise produces noisy geometry. |
| BookCorpus | Copyright status murky for post-2000 books. Gutenberg is legally cleaner. |
| Any dataset with only `input_ids` field | Pre-tokenised — violates NO TOKENS. |
| Any "embeddings" dataset | Pre-computed weight vectors — exactly what we are replacing. |
| News corpora (RealNews, CC-News) | Recency-biased; heavy proper nouns that do not generalise across the manifold. |

### 7.4  Recommended Build Order

1. **Wikipedia** — establishes bulk geometry across all 16 domains; builds ~70K word vocabulary
2. **OpenWebText** — adds informal register; fills casual and hedging vocabulary gaps
3. **SNLI + MultiNLI** — high-signal direct C4 input; sharpens logical fiber immediately
4. **Gutenberg** — enriches narrative and temporal causal language; populates Level 3 sentence frames
5. **SciCite / S2ORC** — only if causal fiber alignment metric (§6.2 target >0.65) is not met after step 4

Steps 1–4 alone comfortably reach the 100K vocabulary target. Each dataset is raw text in, co-occurrence arithmetic in the middle, geometry out the other side — no model is involved at any step.

---

## 8. What This Specification Does NOT Do

The following are explicitly **out of scope** and would violate the design constraints:

| Excluded approach | Why excluded |
|---|---|
| Word2Vec / GloVe / FastText | Neural weights — violates NO WEIGHTS |
| BPE / SentencePiece tokenisation | Token IDs — violates NO TOKENS |
| Pre-trained language model embeddings | Trained parameters — violates NO WEIGHTS + NO TRAINING |
| K-means post-hoc clustering | Changes geometry after the fact without causal structure — violates CAUSALITY FIRST |
| Attention-weighted phrase composition | Learned attention = weights — violates NO WEIGHTS |
| Storing PMI matrix permanently | The PMI matrix is evidence used to generate C4 judgments, then discarded. Retaining it as a lookup table would bypass the manifold and violate SEPARATION. |

---

## 9. Open Questions

The following require answers before Phase 7a implementation begins. They are **not** blockers for writing code, but they affect parameter choices.

1. **Temperature during vocabulary placement.** Should words be placed at `T_floor` (conservative) or at a dedicated `T_vocab` slightly above floor? Higher temperature during initial placement allows words to settle more freely, but risks displacing existing seed concepts.

2. **Isolation of vocabulary words from existing seed geometry.** The 104D manifold already contains 81 seed points (`causal::*`, `prob::*`, `logical::*`, `domain::*`). Should vocabulary words (`vocab::*`) be placed in a dedicated region of the similarity fiber (e.g., dims 32–63 reserved) to avoid displacing the seed geometry? Or should they coexist freely, relying on density crystallisation to protect established regions?

3. **Corpus selection.** Which public-domain corpus or corpora best span the 16-domain universal taxonomy already encoded in the similarity fiber? A corpus that maps poorly to the taxonomy will place words in geometrically sparse regions where they cannot be distinguished by the existing metric.

4. **Phrase generation strategy.** Level 2 (phrase entries, §4.4) requires identifying "nearby words" by geodesic radius. The correct radius threshold depends on how dense the vocabulary region of the manifold becomes after 100K placements. This cannot be known until Phase 7b is complete. Phase 7c should therefore include a calibration step that measures cluster density before setting the phrase-combination threshold.

5. **Streaming vs. batch for production.** The spec describes an offline vocabulary build producing `vocabulary.npz`. For the long-term goal of continuous growth, the `ContrastScheduler` should be able to run as a background process that updates the vocabulary store in-place as new text is encountered. This streaming mode is not blocking for Phase 7, but the `VocabularyStore` serialisation format should be designed to support incremental append without full rewrite.

---

## 10. Relationship to Other Components

This specification touches only three existing files and adds one new module:

| Component | Change |
|---|---|
| `src/phase1/expression/matcher.py` | Add `load_vocabulary()` — 10 lines of code |
| `src/phase3/annealing_engine/engine.py` | Called by `WordPlacer` — no changes required |
| `src/phase2/contrast_engine/engine.py` | Called by `ContrastScheduler` — no changes required |
| `src/vocabulary/` | **New module** — all new code lives here |

All existing tests (516/516) must remain green after Phase 7 is complete. The vocabulary loading is additive — it does not change any existing behaviour when `load_vocabulary()` is not called.

---

## 11. Phase 7 — AI Agent Handoff

This section is written for the AI agent that will implement Phase 7. Read it before reading any source code.

### 11.1  What You Are Continuing

You are implementing Phase 7 of FLOW — a weight-free, token-free reasoning architecture. Six phases are complete. The full C1–C7 pipeline runs end-to-end and produces coherent language output. Your job is to grow the C7 vocabulary from 32 hand-crafted templates to ~100,000 geometrically-grounded entries.

**Before writing a single line of code, read:**
1. `.github/copilot-instructions.md` — system-wide design constraints, API contracts, coding conventions
2. `architecture-specification.md` — the canonical mathematical specification for every component
3. `Phase-6_completed.md` — the most recent completed phase, showing the exact format expected
4. This document in full

### 11.2  Six Design Constraints (Absolute)

Every line you write must satisfy all six:

```
1. NO WEIGHTS      — no tunable parameters; co-occurrence counts are raw statistics, not weights
2. NO TOKENS       — words are geometric events, not symbol IDs; no tokeniser, no BPE, no vocab IDs
3. NO TRAINING     — vocabulary growth is a continuous C3/C4 pass, not a separate offline phase
4. LOCAL UPDATES   — C3 Gaussian falloff must be respected; no global manifold rewrites
5. CAUSALITY FIRST — directed PMI must bias the causal fiber (dims 64–79); causation is structural
6. SEPARATION      — src/vocabulary/ is Group A (manifold shaping) only; never call C5/C6/C7 from it
```

If you are ever tempted to use `Word2Vec`, `GloVe`, `FastText`, any HuggingFace model/pipeline, `transformers`, `torch`, `tensorflow`, `sklearn`, or `gensim` — stop. Those violate constraint 1 or 2. The only permitted libraries are `numpy`, `scipy`, `networkx`, and `datasets` (for raw text streaming only).

### 11.3  File Layout to Create

```
src/vocabulary/
├── __init__.py               # exports VocabularyBuilder, VocabularyLoader
├── cooccurrence.py           # CoOccurrenceCounter
├── word_placer.py            # WordPlacer
├── contrast_scheduler.py     # ContrastScheduler
├── template_builder.py       # TemplateBuilder
├── vocabulary_store.py       # VocabularyStore
└── builder.py                # VocabularyBuilder (orchestrator)

src/phase1/expression/matcher.py   # ADD load_vocabulary() only; do not change any existing method

tests/test_phase7a.py          # Phase 7a unit tests
tests/test_phase7b.py          # Phase 7b unit tests
tests/test_phase7c.py          # Phase 7c unit tests
tests/test_phase7.py           # Full Phase 7 integration tests
tests/phase-7_demo.py          # End-to-end demo script
Phase-7_completed.md           # Completion record (write last)
```

### 11.4  Key API Contracts to Call (Do Not Reimplement)

These already exist. Import and use them:

```python
# Place a word on the manifold via C3
from src.phase3.annealing_engine import AnnealingEngine, Experience
result = annealing_engine.process(Experience(vector=vec_104d, label="vocab::word"))

# Apply a SAME/DIFFERENT judgment via C4
from src.phase2.contrast_engine import ContrastEngine
from src.phase2.contrast_engine.engine import JudgmentType
result = contrast_engine.judge("vocab::w1", "vocab::w2", JudgmentType.SAME, strength=0.7)

# Read a word's 104D position after placement
position = manifold.position("vocab::word")   # np.ndarray shape (104,)

# Read manifold density at a position
density = manifold.density(position)           # float

# Load vocabulary into matcher
from src.phase1.expression import ExpressionRenderer
renderer = ExpressionRenderer()
renderer.matcher.load_vocabulary("vocabulary.npz")  # method you will add
```

### 11.5  New API Contracts to Implement

```python
# CoOccurrenceCounter
from src.vocabulary import CoOccurrenceCounter
counter = CoOccurrenceCounter(window_size=5, min_count=5, v_max=100_000)
counter.feed("plain text string")      # process one text; can be called repeatedly
counter.feed_stream(iterable_of_str)   # convenience: calls feed() for each item
matrix = counter.build()               # CoOccurrenceMatrix — call once when stream is done
matrix.pmi("word1", "word2")          # float — symmetric PMI
matrix.dpmi("word1", "word2")         # float — directed PMI (w1→w2)
matrix.vocabulary                      # list[str] — all words seen above min_count

# WordPlacer
from src.vocabulary import WordPlacer
placer = WordPlacer(annealing_engine)
placer.place("acceleration", freq_rank=1234)   # places vocab::acceleration on M(t)

# ContrastScheduler
from src.vocabulary import ContrastScheduler
scheduler = ContrastScheduler(
    contrast_engine,
    tau_same=1.0,
    tau_diff=-0.5,
    batch_size=256,
)
n_judgments = scheduler.run(matrix)   # int — total judgments applied

# TemplateBuilder
from src.vocabulary import TemplateBuilder
builder = TemplateBuilder(manifold)
entries = builder.build(matrix)       # list[ExpressionEntry]

# VocabularyStore
from src.vocabulary import VocabularyStore
VocabularyStore.save(entries, "vocabulary.npz")
entries = VocabularyStore.load("vocabulary.npz")   # list[ExpressionEntry]

# VocabularyBuilder (orchestrator)
from src.vocabulary import VocabularyBuilder
vbuilder = VocabularyBuilder(
    manifold,
    annealing_engine,
    contrast_engine,
    window_size=5,
    min_count=5,
    v_max=100_000,
    tau_same=1.0,
    tau_diff=-0.5,
    batch_size=256,
)
vbuilder.feed(text_or_stream)         # ingest text
n_entries = vbuilder.build_and_save("vocabulary.npz")   # int — entries written

# Matcher extension
from src.phase1.expression import ExpressionRenderer
renderer = ExpressionRenderer()
n = renderer.matcher.load_vocabulary("vocabulary.npz")  # int — entries loaded
```

### 11.6  Implementation Sequence

Follow the four-sub-phase order. Do not jump ahead. All prior tests must pass at the end of every sub-phase.

**Phase 7a — Co-occurrence + Word Placement**
1. Implement `CoOccurrenceCounter` in `src/vocabulary/cooccurrence.py`
   - Sliding window (default size 5) over normalised word stream
   - Symmetric count matrix + directed count matrix in memory (`collections.Counter`)
   - `build()` computes PMI and dPMI from counts; discards raw counts
   - `normalise()`: lowercase, strip punctuation with `re.sub(r"[^a-z\s]", "", s)`, split on whitespace — nothing more
2. Implement `WordPlacer` in `src/vocabulary/word_placer.py`
   - `structural_feature_vector(word, freq_rank)` returns a 104D numpy array (see §2.2 for the fiber-by-fiber spec)
   - Calls existing `AnnealingEngine.process()` with `T = T_floor` — do not change the engine
3. Write `tests/test_phase7a.py` with `TestCoOccurrenceCounter` and `TestWordPlacer` classes
4. Write `tests/phase-7a_demo.py` — place 1,000 words from a 10-sentence sample text; print their manifold domain assignments
5. **Acceptance check:** function words (`"the"`, `"a"`, `"is"`) land in probabilistic fiber region with density > 0.6; negation words (`"not"`, `"never"`) have non-zero logical fiber components

**Phase 7b — Contrast Pass**
1. Implement `ContrastScheduler` in `src/vocabulary/contrast_scheduler.py`
   - Iterates PMI pairs, applies `τ_same` / `τ_diff` thresholds → `JudgmentType`
   - Strength = `min(1.0, |pmi| / pmi_max)` (see §2.3)
   - Processes judgments in batches of `batch_size`; calls `manifold.update_density()` between batches
   - Directed dPMI: if `dpmi(w1→w2) > dpmi(w2→w1) + 0.5`, apply a displacement biased into causal fiber dims 64–79
2. Write `tests/test_phase7b.py` with `TestContrastScheduler`
3. **Acceptance check:** after 3 passes of contrast judgments on a 1,000-word vocabulary — `manifold.distance("vocab::fast", "vocab::rapid") < manifold.distance("vocab::fast", "vocab::slow")`

**Phase 7c — Template Construction + Store**
1. Implement `TemplateBuilder` in `src/vocabulary/template_builder.py` (three levels, §4.4)
2. Implement `VocabularyStore` in `src/vocabulary/vocabulary_store.py` (numpy .npz format, §4.5)
3. Add `load_vocabulary(path: str) -> int` to `ResonanceMatcher` in `src/phase1/expression/matcher.py`
   - Load entries from `.npz`, append to `self.vocabulary` (do not replace existing 32 entries)
   - Return count of entries loaded
4. Write `tests/test_phase7c.py` with `TestTemplateBuilder`, `TestVocabularyStore`, `TestMatcherLoad`
5. **Acceptance check:** load 10,000 entries; call `matcher.match(segment)` — result's `expression.text` is drawn from a vocabulary-derived entry, not a hand-crafted template

**Phase 7d — Orchestration, Scale, and Integration**
1. Implement `VocabularyBuilder` orchestrator in `src/vocabulary/builder.py`
2. Write `src/vocabulary/__init__.py` exporting `VocabularyBuilder`, `CoOccurrenceCounter`, `WordPlacer`, `ContrastScheduler`, `TemplateBuilder`, `VocabularyStore`
3. Write `tests/test_phase7.py` — full integration tests; confirm all 516 prior tests still pass
4. Write `tests/phase-7_demo.py` following the exact format of `tests/phase-6_demo.py`:
   - Build pipeline, ingest sample text through VocabularyBuilder, save vocabulary
   - Load vocabulary into matcher, run 4 pipeline queries
   - Print: entry count, template diversity ratio, example rendered output, evaluation suite results
5. Write `Phase-7_completed.md` following the format of `Phase-6_completed.md`
6. **Acceptance check:** template diversity ratio ≥ 0.8 across a 10-query evaluation suite; vocabulary coverage (§6.4) > 0.90

### 11.7  Testing Pattern

Follow the exact pattern of every prior phase:

```python
# tests/test_phase7a.py
class TestCoOccurrenceCounter:
    def test_feed_basic(self): ...
    def test_pmi_symmetric(self): ...
    def test_dpmi_directed(self): ...
    def test_min_count_pruning(self): ...
    def test_vocabulary_ceiling(self): ...
    # aim for ~25 tests per class

class TestWordPlacer:
    def test_place_function_word(self): ...
    def test_place_negation_word(self): ...
    def test_place_content_word(self): ...
    def test_label_format(self): ...   # must be "vocab::{word}"
    # aim for ~20 tests
```

Run the full suite after each sub-phase:
```bash
PYTHONPATH=/Users/admin/Desktop/FLOW .venv/bin/python -m pytest tests/ --tb=short -q
```

All prior tests (516+) must be green before moving to the next sub-phase.

### 11.8  Answers to Open Questions (From §9)

These are resolved decisions — do not re-open them:

1. **Temperature during vocabulary placement:** Use `T_floor` (cold). Words land conservatively close to their structural initial position; contrast refinement does the fine-grained work. This protects the 81 existing seed concepts from displacement.

2. **Isolation of vocabulary words:** Coexist freely with seed geometry. The density crystallisation mechanism already in production (C3 `RegionClassifier`) will protect crystallised seed regions. Vocabulary words in the `vocab::` namespace do not carry the reserved-region flags that seed concepts do, so they settle naturally without special enforcement.

3. **Corpus selection:** Wikipedia + OpenWebText + SNLI/MultiNLI + Gutenberg (build order in §7.4). Provides full 16-domain coverage.

4. **Phrase generation threshold:** Phase 7c must include a calibration step before setting the geodesic radius threshold for Level 2 phrases: after vocabulary placement, compute the mean pairwise distance within the top-10 densest clusters on the similarity fiber; set phrase-combination radius = 0.5 × that mean distance.

5. **Streaming vs batch:** Phase 7 delivers offline batch build to `vocabulary.npz`. The `VocabularyStore` serialisation format must support incremental append (open with `'a'` mode or maintain a staging buffer) so a future Phase 8 streaming mode requires no format migration.

### 11.9  What Success Looks Like

When Phase 7 is complete, `GEOPipeline.query()` output will contain sentences like:

```
"The rapid acceleration of the underlying mechanism reveals that propagation
through the causal pathway is consistent with what the evidence suggests."
```

Instead of the current:

```
"This demonstrates that mechanism 4. Unlike mechanism 0 — specifically
regarding mechanism 0 and mechanism 1 —, the mechanism 1 is different in
that their interaction."
```

The words `"rapid"`, `"acceleration"`, `"reveals"`, `"propagation"`, `"consistent"`, `"evidence"` are all manifold concepts with 104D positions derived from co-occurrence geometry — not hard-coded strings, not neural embeddings, not token predictions. They are shapes on the manifold that happen to have English names.
