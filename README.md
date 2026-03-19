# FLOW — Geometric Causal Architecture

**Phase 1 Implementation**

This repository implements Phase 1 of the Geometric Causal Architecture described in the specification documents.  It is a weight-free, token-free reasoning architecture that stores knowledge as shape in high-dimensional space.

---

## Design Principles

```
1. NO WEIGHTS      — knowledge is geometric shape, not tunable parameters
2. NO TOKENS       — meaning is continuous trajectory, not symbol sequences
3. NO TRAINING     — growth is the operating mode, not a separate phase
4. LOCAL UPDATES   — new knowledge deforms nearby geometry only
5. CAUSALITY FIRST — cause-effect relationships are encoded structurally
6. SEPARATION      — meaning generation and language expression are separate
```

---

## Phase 1 — What Is Built

### Phase 1a — Seed Geometry Engine (Component 1)

The Seed Geometry Engine derives `M₀` — the initial mathematical skeleton of the manifold — from first principles.  **No data.  No training.  Runs once.**

Four base geometries are composed into a unified 104-dimensional manifold via fiber bundle construction:

| Geometry | Source | Dimension | Encodes |
|---|---|---|---|
| Similarity | Metric space axioms | 64D | Conceptual distance, domain structure |
| Causal | Pearl's do-calculus | 16D | Causal direction, interventional structure |
| Logical | Boolean algebra | 8D | Contradiction, entailment, negation |
| Probabilistic | Kolmogorov + Fisher | 16D | Confidence, uncertainty gradients |

**Total: 104 dimensions.**

The fiber bundle composition:
- Similarity metric = the base manifold (the "floor")
- Causal structure = directionality fiber (the "flow")
- Logical topology = constraint fiber (the "walls")
- Probabilistic simplex = confidence fiber (the "light")

### Phase 1b — Expression Renderer (Component 7 prototype)

The Expression Renderer converts a standing wave `Ψ` into fluent natural language.  It has **no access to the manifold** — it receives only `Ψ`.

Three-stage pipeline:
1. **Segmentation** — find natural meaning boundaries from amplitude structure
2. **Resonance matching** — constraint satisfaction: find the expression whose semantic wave best matches each segment
3. **Flow preservation** — adjust language to reflect the trajectory's dynamics (fast flow → short sentences, uncertainty → hedging, etc.)

---

## Directory Structure

```
FLOW/
├── architecture-specification.md     # Full system spec
├── geometric-causal-architecture.md  # Conceptual overview
├── manifold-initialization.md        # Initialization theory
├── language-output-without-tokens.md # Token-free output theory
├── requirements.txt
│
└── src/
    └── phase1/
        ├── seed_geometry/
        │   ├── causal.py         # CausalGeometry — Pearl's do-calculus
        │   ├── logical.py        # LogicalGeometry — Boolean hypercube
        │   ├── probabilistic.py  # ProbabilisticGeometry — Fisher metric
        │   ├── similarity.py     # SimilarityGeometry — base manifold
        │   ├── composer.py       # FiberBundleComposer — composition
        │   ├── manifold.py       # SeedManifold (M₀) — query API
        │   └── engine.py         # SeedGeometryEngine — public entry point
        │
        └── expression/
            ├── wave.py           # StandingWave + mock wave constructors
            ├── matcher.py        # ResonanceMatcher — constraint satisfaction
            └── renderer.py       # ExpressionRenderer — full pipeline
```

---

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Build M₀

```python
from src.phase1.seed_geometry import SeedGeometryEngine

engine = SeedGeometryEngine()
M0 = engine.build()

# Query the manifold
p = M0.seed_points[0]
q = M0.seed_points[1]

print(f"Distance: {M0.distance(p, q):.4f}")
print(f"Domain of p: {M0.domain_of(p)}")
print(f"Confidence at p: {M0.confidence(p):.3f}")
print(f"Causal ancestry (p→q): {M0.causal_ancestry(p, q)}")
```

### Render a Standing Wave

```python
from src.phase1.expression import ExpressionRenderer, create_mock_wave

# Create a mock standing wave (normally produced by the Resonance Layer)
wave = create_mock_wave("explanation")

# Render to natural language
renderer = ExpressionRenderer()
output   = renderer.render(wave)

print(output.text)
print(f"Rendering confidence: {output.confidence:.3f}")
```

### End-to-End Demo

```python
from src.phase1.seed_geometry import SeedGeometryEngine
from src.phase1.expression import ExpressionRenderer, create_mock_wave

# Phase 1a: Build the seed manifold
engine = SeedGeometryEngine()
M0     = engine.build()

# Phase 1b: Render a wave (mock — normally from Phase 4's Resonance Layer)
for theme in ["causation", "uncertainty", "contrast"]:
    wave   = create_mock_wave(theme)
    output = ExpressionRenderer().render(wave)
    print(f"\n[{theme.upper()}]")
    print(output.text)
```

---

## Run Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src

# Specific test class
python -m pytest tests/test_phase1.py::TestSeedManifold -v
```

---

## Build Order (from architecture spec)

```
PHASE 1 — FOUNDATIONS          ← YOU ARE HERE
  1a. Seed Geometry Engine      ✓ DONE
  1b. Expression Renderer       ✓ DONE (mock wave input)

PHASE 2 — THE MANIFOLD         (next)
  2a. Living Manifold data structures  (research problem)
  2b. Contrast Engine                  (runs on Phase 2a)

PHASE 3 — SHAPING
  3a. Annealing Engine                 (runs on Phase 2a)

PHASE 4 — REASONING
  4a. Flow Engine                      (SDE on manifold)
  4b. Resonance Layer                  (wave accumulation)

PHASE 5 — INTEGRATION
  5a. Full pipeline end-to-end
  5b. New evaluation framework
```

---

## Mathematical Foundations

| Component | Mathematics |
|---|---|
| Causal geometry | Riemannian geometry, Pearl's do-calculus, DAG embeddings |
| Logical geometry | Boolean algebra, hypercube topology |
| Probabilistic geometry | Information geometry (Amari), Fisher-Rao metric, Shannon entropy |
| Similarity geometry | Riemannian manifold theory, metric space axioms |
| Composition | Fiber bundle theory (Steenrod) |
| Flow engine (Phase 4) | Stochastic differential geometry, SDE on Riemannian manifolds |
| Resonance layer (Phase 4) | Wave mechanics on manifolds |

---

## Key Properties Implemented

- **No weights anywhere** — zero tunable numerical parameters
- **No tokenization** — language output from continuous wave, not symbol prediction
- **Metric PSD guaranteed** — validated at build time
- **Causal asymmetry** — retro-causal travel costs more by construction
- **Locality guarantee** — deformation operations are local by design
- **Confidence encoding** — density and probability fiber encode certainty
- **Language agnosticism** — swap the `ResonanceMatcher` vocabulary for any language
