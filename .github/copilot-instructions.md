# GitHub Copilot Instructions — FLOW

## What We Are Building

**FLOW** is a weight-free, token-free reasoning architecture that stores knowledge as shape in high-dimensional geometric space.  There are no weight matrices, no tokenisers, no gradient descent, and no training phase.  The system reasons by navigating a Riemannian manifold and expresses results by rendering a standing wave into natural language.

The full system is called the **Geometric Causal Architecture**.  It has seven numbered components arranged in two logical groups:

```
Group A — The Manifold (What The System Knows)
  C1  Seed Geometry Engine   — derives M₀ from first principles (runs once, static forever)
  C2  Living Manifold        — dynamic M(t), full READ + WRITE API
  C3  Annealing Engine       — continuous self-organisation of M(t)
  C4  Contrast Engine        — SAME/DIFFERENT relational judgments that shape geometry

Group B — The Reasoner (What The System Does)
  C5  Flow Engine            — navigates M(t) via SDE to produce meaning as trajectory
  C6  Resonance Layer        — accumulates trajectory into pre-linguistic standing wave Ψ
  C7  Expression Renderer    — converts Ψ into fluent natural language (no token prediction)
```

Data flow:
```
Experience → [C1 once] → [C2: Living Manifold] ← [C3 Annealing] ← raw experience
                               │               ← [C4 Contrast]  ← paired experience
                          Query in
                               ↓
                          [C5 Flow Engine]
                               ↓
                          [C6 Resonance Layer]
                               ↓
                          [C7 Expression Renderer]
                               ↓
                          Language out
```

---

## Six Non-Negotiable Design Constraints

Every line of code must be evaluated against all six:

```
1. NO WEIGHTS      — no tunable numerical parameters; knowledge is geometric shape
2. NO TOKENS       — no tokeniser, no token IDs; meaning is a continuous trajectory
3. NO TRAINING     — growth is the operating mode, not a separate phase
4. LOCAL UPDATES   — new knowledge deforms nearby geometry only; distant geometry is never touched
5. CAUSALITY FIRST — cause-effect relationships are encoded structurally in the fiber geometry
6. SEPARATION      — each component does exactly one thing; meaning and language are separate
```

---

## Mathematical Foundations

| Domain | Mathematics Used |
|---|---|
| Causal geometry | Riemannian geometry, Pearl's do-calculus, DAG spectral embeddings |
| Logical geometry | Boolean algebra, Boolean hypercube topology |
| Probabilistic geometry | Information geometry (Amari), Fisher-Rao metric, KL/JS divergence |
| Similarity geometry | Metric space axioms, variable-curvature Riemannian manifold |
| Composition | Fiber bundle theory (Steenrod) |
| Flow Engine (Phase 4) | Stochastic differential geometry, SDE on Riemannian manifolds |
| Resonance Layer (Phase 4) | Wave mechanics on manifolds |

**Manifold dimension: 104D**
- Dims   0–63  : Similarity base manifold (16-domain universal taxonomy)
- Dims  64–79  : Causal fiber (Pearl do-calculus, asymmetric metric γ=2.0)
- Dims  80–87  : Logical fiber (Boolean hypercube, Hamming + continuous distance)
- Dims  88–103 : Probabilistic fiber (Fisher-Rao metric, √-space slerp geodesics)

---

## Environment Setup

```bash
# Python virtual environment lives at .venv/
source .venv/bin/activate

# Dependencies (no ML frameworks — pure mathematics)
pip install -r requirements.txt
```

`requirements.txt` contents:
```
numpy>=1.24.0
scipy>=1.11.0
networkx>=3.1
pytest>=7.4.0
pytest-cov>=4.1.0
```

No PyTorch, TensorFlow, or any weight-based ML library is ever added to this project.

---

## Codebase Structure

```
FLOW/
├── .github/
│   └── copilot-instructions.md        ← this file
│
├── architecture-specification.md      # Full component spec (source of truth)
├── geometric-causal-architecture.md   # Conceptual overview
├── manifold-initialization.md         # M₀ initialization theory
├── language-output-without-tokens.md  # Token-free output theory
│
├── Phase-1_completed.md               # Phase 1 sign-off record
├── Phase-2_completed.md               # Phase 2 sign-off record
├── Phase-3_completed.md               # Phase 3 sign-off record
├── Phase-4_completed.md               # Phase 4 sign-off record
├── Phase-5_completed.md               # Phase 5 sign-off record
├── README.md
├── requirements.txt
│
├── src/
│   ├── phase1/
│   │   ├── seed_geometry/             # C1 — Seed Geometry Engine
│   │   │   ├── causal.py             # CausalGeometry — 16D, Pearl do-calculus
│   │   │   ├── logical.py            # LogicalGeometry — 8D Boolean hypercube
│   │   │   ├── probabilistic.py      # ProbabilisticGeometry — 16D Fisher-Rao
│   │   │   ├── similarity.py         # SimilarityGeometry — 64D base manifold
│   │   │   ├── composer.py           # FiberBundleComposer — 104D unified bundle
│   │   │   ├── manifold.py           # SeedManifold M₀ + ManifoldPoint
│   │   │   └── engine.py             # SeedGeometryEngine — public entry point
│   │   │
│   │   └── expression/               # C7 — Expression Renderer (Phase 1 prototype)
│   │       ├── wave.py               # StandingWave Ψ, WavePoint, WaveSegment, mock factories
│   │       ├── matcher.py            # ResonanceMatcher — 30+ vocab entries, cosine match
│   │       └── renderer.py           # ExpressionRenderer — 3-stage pipeline
│   │
│   ├── phase2/
│   │   ├── living_manifold/           # C2 — Living Manifold
│   │   │   ├── state.py              # DeformationField φ(t), DensityField ρ(t), ManifoldState
│   │   │   ├── regions.py            # RegionClassifier — CRYSTALLIZED/FLEXIBLE/UNKNOWN
│   │   │   ├── geodesic.py           # GeodesicComputer — kNN graph, Dijkstra paths
│   │   │   ├── deformation.py        # LocalDeformation — Gaussian-weighted displacement kernel
│   │   │   └── manifold.py           # LivingManifold — central Phase 2 data structure
│   │   │
│   │   └── contrast_engine/          # C4 — Contrast Engine
│   │       ├── persistence.py        # PersistenceDiagram — birth/death cluster tracking
│   │       └── engine.py             # ContrastEngine — SAME/DIFFERENT judgments
│   │
   ├── phase3/
   │   └── annealing_engine/          # C3 — Annealing Engine
   │       ├── schedule.py           # TemperatureSchedule — T(t) = T₀·e^(−λt) + T_floor
   │       ├── novelty.py            # NoveltyEstimator — distance + density novelty scoring
   │       ├── experience.py         # Experience, ExperienceResult — raw experience data types
   │       └── engine.py             # AnnealingEngine — 5-step experience processing loop
   │
   ├── phase4/
   │   ├── flow_engine/               # C5 — Flow Engine
   │   │   ├── query.py              # Query, FlowStep, Trajectory — input/output data types
   │   │   ├── forces.py             # ForceComputer — 4 geometric drift forces
   │   │   ├── sde.py                # SDESolver — Euler-Maruyama on Riemannian manifold
   │   │   └── engine.py             # FlowEngine — LOCATE→ORIENT→INTEGRATE→TERMINATE
   │   │
   │   └── resonance_layer/           # C6 — Resonance Layer
   │       ├── accumulator.py        # ExcitationKernel, HarmonicKernel, ResonanceAccumulator
   │       └── layer.py              # ResonanceLayer — trajectory → StandingWave Ψ
   │
   └── phase5/
       ├── pipeline/                  # Full system integration
       │   ├── pipeline.py           # GEOPipeline — single C1–C7 entry point
       │   └── result.py             # PipelineResult — bundled query output
       │
       └── evaluation/               # Geometry-grounded evaluation framework
           ├── metrics.py            # CoherenceMetrics, CausalMetrics, LocalityMetrics
           ├── evaluator.py          # PipelineEvaluator — 4 evaluation tasks + run_suite()
           └── suite.py              # SuiteResult — aggregate suite output
│
├── persistence/
│   └── snapshot.py                   # ManifoldSnapshot — .npz save/load for M(t)
│
└── tests/
    ├── test_phase1.py                 # 90 unit tests for Phase 1 (all passing)
    ├── test_phase2.py                 # 95 unit tests for Phase 2 (all passing)
    ├── test_phase3.py                 # 90 unit tests for Phase 3 (all passing)
    ├── test_phase4.py                 # 113 unit tests for Phase 4 (all passing)
    ├── test_phase5.py                 # 128 unit tests for Phase 5 (all passing)
    ├── test_phase8.py                 # 48 unit tests for Phase 8 (47 passing + 1 FAISS skip)
    ├── phase-1_demo.py               # End-to-end Phase 1 demo script
    ├── phase-2_demo.py               # End-to-end Phase 2 demo script
    ├── phase-3_demo.py               # End-to-end Phase 3 demo script
    ├── phase-4_demo.py               # End-to-end Phase 4 demo script
    ├── phase-5_demo.py               # End-to-end Phase 5 demo script
    └── phase-8_demo.py               # End-to-end Phase 8 demo script
```

---

## Phase Build Order and Status

### Phase 1 — COMPLETED ✅ (90/90 tests | demo: 0.013s)

| Component | Module | Status |
|---|---|---|
| C1 — Seed Geometry Engine | `src/phase1/seed_geometry/` | ✅ Done |
| C7 — Expression Renderer (prototype) | `src/phase1/expression/` | ✅ Done (mock wave input) |

**What Phase 1 delivered:**
- `SeedGeometryEngine.build()` → derives 4 geometries → fiber bundle composition → 81 seed points → validates M₀ (104D)
- `SeedManifold` (M₀) with full READ interface: `position()`, `distance()`, `causal_direction()`, `curvature()`, `density()`, `neighbors()`, `nearest()`, `domain_of()`, `locality_radius()`, `causal_ancestry()`, `confidence()`, `logic_certainty()`
- `StandingWave` Ψ data type with `WavePoint`, `WaveSegment`, `create_mock_wave(theme)` factory
- `ExpressionRenderer` 3-stage pipeline: segmentation → resonance matching → flow preservation

### Phase 2 — COMPLETED ✅ (185/185 tests | demo: 1.23s)

| Component | Module | Status |
|---|---|---|
| C2 — Living Manifold | `src/phase2/living_manifold/` | ✅ Done |
| C4 — Contrast Engine | `src/phase2/contrast_engine/` | ✅ Done |

**What Phase 2 delivered:**
- `LivingManifold` with full READ + WRITE API wrapping M₀
  - READ: all M₀ queries plus `geodesic()`, `geodesic_distance()`, `region_type()`
  - WRITE: `place()` (add/move concept), `deform_local()` (Gaussian kernel displacement, returns n_affected), `update_density()`
- `RegionClassifier` mapping density → `CRYSTALLIZED / FLEXIBLE / UNKNOWN` with derived stiffness, locality radius (`r = r_max · exp(−ρ·3)`), diffusion scale
- `GeodesicComputer` with kNN graph and Dijkstra-based shortest path; incremental `add_point` / `update_point`
- `LocalDeformation` Gaussian-weighted kernel — locality guarantee: effect → 0 at distance; centre always moves at full weight
- `ContrastEngine` SAME → pulls P₁/P₂ closer by α; DIFFERENT → pushes apart by β; returns `ContrastResult` with Δdist and displacement vectors
- `PersistenceDiagram` tracks pairwise distances over time, detects birth/death events, proposes cluster corrections

### Phase 3 — COMPLETED ✅ (275/275 tests | demo: 2.63s)

| Component | Module | Status |
|---|---|---|
| C3 — Annealing Engine | `src/phase3/annealing_engine/` | ✅ Done |

**What Phase 3 delivered:**
- `TemperatureSchedule` — `T(t) = T₀·e^(−λt) + T_floor`; `step()` advances internal clock; `reset()` restores T₀; `locality_radius(base, t)` scales deformation radius with temperature; `is_cold()` predicate
- `NoveltyEstimator` — scores how surprising an experience is: `distNovelty = 1 − exp(−d/σ)` (absolute scale) + `densNovelty = 1 − ρ(P)`; weighted combination (0.6/0.4); `consistency_gradient()` returns unit-normalised pull toward neighbour centroid
- `Experience` data type: 104D vector + optional label + source tag
- `ExperienceResult` data type: full processing record (anchor, novelty, temperature, `|δ|`, n_affected, placed_label)
- `AnnealingEngine` — 5-step loop: LOCATE → NOVELTY → DEFORM → APPLY → DENSITY; `process()`, `process_batch()`, `reset_temperature()`, `summary()`; running `AnnealingStats`

### Phase 4 — COMPLETED ✅ (388/388 tests | demo: 1.01s)

| Component | Module | Status |
|---|---|---|
| C5 — Flow Engine | `src/phase4/flow_engine/` | ✅ Done |
| C6 — Resonance Layer | `src/phase4/resonance_layer/` | ✅ Done |

**What Phase 4 delivered:**
- `FlowEngine` — entry point `flow(query) → Trajectory`; 4-step loop: LOCATE (kNN anchor) → ORIENT (initial velocity toward attractor) → INTEGRATE (Euler-Maruyama SDE) → TERMINATE (4 conditions: attractor_reached / revisit_detected / velocity_threshold / max_steps)
- `ForceComputer` — 4 geometric drift forces: (1) Semantic Gravity `Σᵢ mᵢ·(Pᵢ−P)/‖Pᵢ−P‖²`, (2) Causal Curvature `κ(P)·causal_dir`, (3) Contextual Momentum `γ·V_prev` (γ=0.85), (4) Contrast Repulsion `−Σⱼ contr(P,Pⱼ)·(Pⱼ−P)`
- `SDESolver` — Euler-Maruyama `dP = μdt + σdW`; diffusion `σ(P) = scale·(1−density(P))` (crystallised regions precise, sparse regions explorative)
- `Query`, `FlowStep`, `Trajectory` data types; `Trajectory` exposes `positions`, `velocities`, `mean_speed`, `mean_curvature`
- `ResonanceLayer` — entry point `accumulate(trajectory) → StandingWave`; excitation kernel `A·exp(−‖Q−P‖²/2r_eff²)` + harmonic kernel on integer frequency ratios; outputs normalised Ψ ∈ [0,1] → `WavePoints` → `StandingWave` (C7-compatible)
- `ResonanceAccumulator`, `ExcitationKernel`, `HarmonicKernel` sub-components
- Full pipeline C5 → C6 → C7 working end-to-end

### Phase 5 — COMPLETED ✅ (516/516 tests | demo: 5.71s)

| Component | Module | Status |
|---|---|---|
| GEOPipeline — Full integration | `src/phase5/pipeline/` | ✅ Done |
| Evaluation framework | `src/phase5/evaluation/` | ✅ Done |

**What Phase 5 delivered:**
- `GEOPipeline` — single entry point wiring all seven components C1–C7; `learn()` (C3), `learn_batch()`, `contrast()` (C4), `query()` (C5→C6→C7); introspection: `temperature`, `stats`, `query_count`, `n_concepts`, `summary()`
- `PipelineResult` — value object from every `query()` call bundling `Trajectory`, `StandingWave` Ψ, and `RenderedOutput`; convenience props: `text`, `confidence`, `wave_confidence`, `n_steps`, `termination_reason`, `flow_preserved`
- C7 now receives a **real** Ψ from C6, not a mock wave — the full C5→C6→C7 pipeline is integrated and verified
- `CoherenceMetrics` — intrinsic quality score (35% render + 25% wave + 20% core + 20% amplitude)
- `CausalMetrics` — compares forward vs backward trajectories via the causal fiber; `causal_score ∈ [0,1]`
- `LocalityMetrics` — runtime verification of the LOCAL-UPDATES-ONLY constraint; `locality_satisfied` bool
- `PipelineEvaluator` — `evaluate_query()`, `evaluate_causal_direction()`, `evaluate_novelty_decay()`, `evaluate_locality()`, `run_suite() → SuiteResult`
- `SuiteResult` — aggregate metrics: `mean_coherence`, `termination_distribution`, `novelty_is_decaying`
- **Minimum viable demonstration confirmed**: accepts 104D vector query → navigates M(t) via SDE → produces standing wave Ψ → renders natural language — zero weight matrices, zero tokenisation, zero training phase

### Phase 6 — COMPLETED ✅ (516/516 tests | demo: runs clean)

| Component | Module | Change |
|---|---|---|
| C6 — Resonance Layer | `src/phase4/resonance_layer/layer.py` | WavePoint labels resolved to nearest manifold concept (k=5 with sliding-window dedup) instead of `flow_t{i}` placeholders |
| C7 — Expression Renderer | `src/phase1/expression/renderer.py` | `_clean_label()` strips domain prefix; used in all template-fill and anaphora sites |
| C7 — Resonance Matcher | `src/phase1/expression/matcher.py` | Template diversity penalty (+0.15 per recency hit); `match_all()` passes 3-step sliding window |
| C7 — Expression Renderer | `src/phase1/expression/renderer.py` | Fallback filler vocabulary expanded from 5 to 8 varied entries |

**What Phase 6 delivered:**
- Output text now contains **real manifold concept names** (`"perturbation"`, `"mechanism"`, `"maximal uncertainty"`) instead of positional placeholders (`"flow t0"`, `"flow t1"`)
- Domain namespace prefixes stripped at render time: `"causal::co_occurrence"` → `"co occurrence"`
- Consecutive sentence segments use **different grammatical structures** (diversity ratio 0.67 vs ~0.17 before)
- All 516 prior tests remain green — Phase 6 enhances behaviour without breaking any contract

### Architecture — COMPLETE ✅

All seven components (C1–C7) are implemented, integrated, and producing coherent language output with real concept names from the manifold.

### Phase 8 — COMPLETED ✅ (721/722 tests | 1 skipped optional FAISS | demo: clean)

| Component | Module | Change |
|---|---|---|
| Persistence | `src/persistence/snapshot.py` | ManifoldSnapshot .npz save/load with perfect round-trip fidelity |
| C2 — Living Manifold | `src/phase2/living_manifold/manifold.py` | KDTree → cKDTree; deformation pre-filter via query_ball_point |
| C2 — Deformation | `src/phase2/living_manifold/deformation.py` | apply() accepts optional candidate_labels pre-filter |
| C2 — Geodesic | `src/phase2/living_manifold/geodesic.py` | Incremental kNN graph update with dirty-label tracking |
| C7 — Matcher | `src/phase1/expression/matcher.py` | Optional FAISS integration for accelerated vocabulary matching |
| Pipeline | `src/phase5/pipeline/pipeline.py` | save() / load() convenience methods |

**What Phase 8 delivered:**
- `ManifoldSnapshot.save(manifold, path)` / `.load(path)` — full M(t) persistence as .npz (perfect 0.00e+00 position error)
- `cKDTree` replaces `KDTree` everywhere (10–50× faster spatial queries)
- Deformation pre-filter via `cKDTree.query_ball_point()` — O(k_local) instead of O(n)
- Incremental geodesic graph — dirty-label tracking with threshold-based rebuild decision
- Optional FAISS `IndexFlatIP` for vocabulary matching at >200 entries (graceful fallback)
- `GEOPipeline.save()` / `GEOPipeline.load()` — single-call pipeline persistence
- All 674 prior tests remain green — Phase 8 is pure scaling, no behavioral change

---

## Per-Phase Implementation Pattern

Every phase follows a strict pattern:

1. **Source of truth**: read `architecture-specification.md` for the component's responsibility, interfaces, and mathematical specification before writing any code.
2. **Module layout**: each component gets its own subdirectory under `src/phaseN/component_name/` with an `__init__.py` that exports the public API.
3. **Tests**: `tests/test_phaseN.py` — collected into named `TestClass` groups, one per class.  All tests for all prior phases must remain green.
4. **Demo**: `tests/phase-N_demo.py` — a standalone runnable script that exercises the public API end-to-end and prints labeled output to stdout.
5. **Completed record**: `Phase-N_completed.md` in the workspace root — records scope, what was built (table of files + purpose), full test results table, demo output verbatim, design constraints upheld table, and issues resolved.

### Running tests

```bash
# All phases
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src

# Single phase
python -m pytest tests/test_phase5.py -v

# Single class
python -m pytest tests/test_phase5.py::TestGEOPipeline -v

# Run a demo
python tests/phase-5_demo.py
```

---

## Key API Contracts (Do Not Break)

### SeedGeometryEngine (C1)
```python
from src.phase1.seed_geometry import SeedGeometryEngine
engine = SeedGeometryEngine()
M0 = engine.build()          # idempotent; returns SeedManifold
M0.dimension                 # 104
len(M0.seed_points)          # 81
```

### SeedManifold / LivingManifold shared READ interface
```python
M.position(label: str) -> np.ndarray          # 104D vector
M.distance(p, q) -> float
M.causal_direction(p, q) -> float             # > 0 means p causes q
M.causal_ancestry(p, q) -> bool
M.curvature(p) -> float
M.density(p) -> float
M.neighbors(p, k=5) -> list[np.ndarray]
M.nearest(p, k=5) -> list[str]
M.domain_of(p) -> str
M.locality_radius(p) -> float
M.confidence(p) -> float
M.logic_certainty(p) -> float
```

### LivingManifold extra READ (C2)
```python
M.geodesic(p, q) -> list[np.ndarray]          # waypoints
M.geodesic_distance(p, q) -> float
M.region_type(p) -> RegionType                 # CRYSTALLIZED | FLEXIBLE | UNKNOWN
```

### LivingManifold WRITE (C2)
```python
M.place(label: str, position: np.ndarray)      # add or move concept
n = M.deform_local(label: str, displacement: np.ndarray, radius: float) -> int  # n_affected
M.update_density(label: str)
```

### ContrastEngine (C4)
```python
from src.phase2.contrast_engine import ContrastEngine
ce = ContrastEngine(manifold)
result = ce.judge(label_a, label_b, relation)  # relation: "same" | "different"
result.delta_distance    # float, negative=closer, positive=farther
result.direction         # "correct" | "incorrect"
```

### AnnealingEngine (C3)
```python
from src.phase3.annealing_engine import AnnealingEngine, Experience
engine = AnnealingEngine(manifold, T0=1.0, lambda_=0.01, T_floor=0.05)
result = engine.process(Experience(vector=vec, label="concept::x"))
result.novelty           # float in [0, 1]
result.delta_magnitude   # float — L2 norm of applied displacement
result.n_affected        # int   — manifold points displaced
result.placed_label      # str | None — label placed on manifold
engine.temperature       # float — current T(t)
engine.stats             # AnnealingStats — running totals
engine.reset_temperature()               # restart schedule; does NOT undo deformations
results = engine.process_batch(experiences)  # list[ExperienceResult]
```

### FlowEngine (C5)
```python
from src.phase4.flow_engine import FlowEngine, Query
engine = FlowEngine(manifold)
traj = engine.flow(Query(vector=vec, label="what causes X?"))
traj.positions           # list[np.ndarray] — 104D waypoints
traj.mean_speed          # float
traj.mean_curvature      # float
len(traj)                # int — number of FlowSteps
traj[-1].termination_reason  # str — why flow stopped
```

### ResonanceLayer (C6)
```python
from src.phase4.resonance_layer import ResonanceLayer
layer = ResonanceLayer(manifold)
wave = layer.accumulate(trajectory)    # trajectory from FlowEngine
wave.points              # list[WavePoint] — Ψ at each site
wave.confidence          # float — overall wave confidence
# wave is a StandingWave — directly compatible with ExpressionRenderer (C7)
```

### ExpressionRenderer (C7)
```python
from src.phase1.expression import ExpressionRenderer, create_mock_wave
wave   = create_mock_wave("causation")   # themes: causation|uncertainty|contrast|explanation|...
output = ExpressionRenderer().render(wave)
output.text          # str — rendered natural language
output.confidence    # float
```

### GEOPipeline (Phase 5 — main entry point)
```python
from src.phase5 import GEOPipeline, PipelineResult

pipeline = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05, flow_seed=42)

# Group A — shape the manifold
pipeline.learn(Experience(vector=vec, label="concept::x"))   # C3
pipeline.learn_batch(experiences)                            # C3 batch
pipeline.contrast("label_a", "label_b", "same")             # C4

# Group B — reason
result = pipeline.query(vec, label="what is X?")             # C5 → C6 → C7
result.text               # str — rendered natural language
result.confidence         # float — C7 rendering confidence
result.wave_confidence    # float — C6 wave quality
result.n_steps            # int — trajectory length
result.termination_reason # str
result.flow_preserved     # bool
result.trajectory         # Trajectory (C5 output)
result.wave               # StandingWave Ψ (C6 output)

# Introspection
pipeline.temperature      # float — current T(t)
pipeline.n_concepts       # int — concepts on M(t)
pipeline.query_count      # int
pipeline.summary()        # str
```

### PipelineEvaluator (Phase 5 — evaluation)
```python
from src.phase5 import PipelineEvaluator, SuiteResult

ev = PipelineEvaluator(pipeline)
ev.evaluate_query(vec, label)               # → EvaluationResult (coherence metrics)
ev.evaluate_causal_direction(c_vec, e_vec)  # → CausalMetrics  (causal_score ∈ [0,1])
ev.evaluate_novelty_decay(vec, n_reps=5)    # → list[float]    (should decrease)
ev.evaluate_locality(vec, label)            # → LocalityMetrics (locality_satisfied bool)
suite = ev.run_suite(vectors, labels)       # → SuiteResult
suite.mean_coherence          # float
suite.novelty_is_decaying     # bool
suite.termination_distribution # dict[str, int]
```

### ManifoldSnapshot (Phase 8 — persistence)
```python
from src.persistence import ManifoldSnapshot

# Save
n = ManifoldSnapshot.save(manifold, "manifold_v1.npz")

# Load (creates fresh LivingManifold from M₀, restores state)
manifold = ManifoldSnapshot.load("manifold_v1.npz")

# Load into existing manifold
ManifoldSnapshot.load("manifold_v1.npz", manifold=existing_manifold)

# Metadata without full load
info = ManifoldSnapshot.info("manifold_v1.npz")
info["n_points"]        # int
info["dimension"]       # 104
info["format_version"]  # 1
```

### GEOPipeline save/load (Phase 8)
```python
from src.phase5 import GEOPipeline

pipeline = GEOPipeline()
# ... learn, contrast, query ...

# Save
pipeline.save("manifold.npz", vocabulary_path="vocab.npz")

# Load
pipeline2 = GEOPipeline.load("manifold.npz", vocabulary_path="vocab.npz", flow_seed=42)
result = pipeline2.query(vec, label="question")
```

---

## Coding Conventions

- All geometry computations use **numpy** and **scipy** only — no ML libraries.
- Vectors passed to manifold methods may be `np.ndarray` **or** `str` (label); methods must handle both.
- Manifold dimension slices: similarity `[0:64]`, causal `[64:80]`, logical `[80:88]`, probabilistic `[88:104]`.
- Deformation operations **must** enforce Gaussian falloff; the centre point always displaces at full weight, but no targeted movement bypasses crystallisation logic for neighbours.
- Each geometry class must validate its output metric is positive semi-definite at construction time.
- `SeedGeometryEngine.build()` must be idempotent (safe to call multiple times).
- New components expose a single clean public entry-point class (e.g. `AnnealingEngine`, `FlowEngine`) imported via the package `__init__.py`.
- Do not add print statements to `src/`; demos and tests handle output.
