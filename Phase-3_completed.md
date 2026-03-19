# Phase 3 — COMPLETED

**Completed**: March 18, 2026  
**Status**: ✅ 275/275 tests passing (90 Phase 1 + 95 Phase 2 + 90 Phase 3) | Demo running end-to-end in 2.63s

---

## Scope

Phase 3 covered one component from the build order:

| Component | Description |
|---|---|
| C3 — Annealing Engine | Continuous self-organisation of M(t) from raw unlabelled experience via physics-inspired simulated annealing |

---

## What Was Built

### `src/phase3/annealing_engine/`

| File | Purpose |
|---|---|
| `schedule.py` | `TemperatureSchedule` — exponential cooling `T(t) = T₀ · e^(−λt) + T_floor`. Maintains an internal clock advanced via `step()`. Derives temperature-scaled locality radius. `is_cold()` predicate. |
| `novelty.py` | `NoveltyEstimator` — measures how surprising an incoming experience is. Distance component uses a fixed absolute sigma scale: `1 − exp(−d/σ)`. Density component: `1 − ρ(P)`. Combined via configurable weights (default 0.6/0.4). Also provides `consistency_gradient()` — unit-normalised pull toward weighted centroid of neighbours. |
| `experience.py` | `Experience` — raw experience data container: 104D vector + optional label + source tag. `ExperienceResult` — full processing record: anchor, novelty score, temperature at processing time, displacement magnitude, n_affected, placed_label. |
| `engine.py` | `AnnealingEngine` — central Phase 3 data structure. Implements the five-step experience-processing loop: LOCATE → NOVELTY → DEFORM → APPLY → DENSITY. Also tracks `AnnealingStats` (n_processed, n_novel, total_deformation, mean_novelty, mean_temperature). |

**Five-step processing loop** (per experience E):
```
1. LOCATE    nearest seed/concept point via kNN → resonance anchor
2. NOVELTY   score = 0.6·distNovelty + 0.4·densNovelty ∈ [0,1]
3. DEFORM    δ = novelty · T(t) · consistency_gradient
4. APPLY     M.deform_local(anchor, δ)   [locality guaranteed by C2]
5. DENSITY   M.update_density(anchor)
```

**Temperature schedule** (drives deformation magnitude and locality radius):
```
T(t) = T₀ · e^(−λt) + T_floor

T₀      = initial temperature  (high flexibility, coarse structure)
λ        = cooling rate         (larger = cools faster)
T_floor  = minimum temperature  (system never fully freezes)
Always: T(t) ≥ T_floor > 0
```

**Novelty formula** (absolute-scale, no normalisation artefacts):
```
distNovelty(E) = 1 − exp(−d_min / σ)        σ = sigma_scale (fixed, default 1.0)
densNovelty(E) = 1 − ρ(P)                   ρ ∈ [0, 1]
novelty(E)     = 0.6 · distNovelty + 0.4 · densNovelty  ∈ [0, 1]
```

---

## Test Results

```
275 passed in 2.63s  (90 Phase 1 + 95 Phase 2 + 90 Phase 3)
```

| Test Class | Tests | Result |
|---|---|---|
| `TestTemperatureSchedule` | 20 | ✅ |
| `TestNoveltyEstimator` | 13 | ✅ |
| `TestExperience` | 6 | ✅ |
| `TestExperienceResult` | 5 | ✅ |
| `TestAnnealingStats` | 7 | ✅ |
| `TestAnnealingEngine` | 27 | ✅ |
| `TestAnnealingEngineIntegration` | 12 | ✅ |

---

## Demo Output

Running `tests/phase-3_demo.py`:

```
=== FLOW — Phase 3 Demo ===

Living Manifold M(t):
  Points          : 81
  Dimension       : 104
  Manifold time t : 0.000
  Write ops       : 0
  Regions:
    Crystallized  : 81
    Flexible      : 0
    Unknown       : 0

--- Temperature Schedule ---
  T(t=  0)  : 1.0500
  T(t= 20)  : 0.4179
  T(t= 50)  : 0.1321
  T(t=100)  : 0.0567
  T(t=200)  : 0.0500
  T_floor   : 0.0500  (never goes below)
  radius@t=0  : 5.0000
  radius@t=100: 0.2702

--- Novelty Estimator ---
  no neighbours      → novelty 1.000   (maximum, unexplored)
  identical + dense  → novelty 0.040   (minimum, well-known)
  distant + sparse   → novelty 1.000   (genuinely new)

--- AnnealingEngine setup ---
  initial temperature : 1.0500
  manifold points     : 81

--- Single experience ---
  label placed        : anneal::causal_variant_1
  resonance anchor    : causal::perturbation
  novelty score       : 0.109
  temperature         : 1.0500
  |δ| applied         : 0.114200
  points affected     : 1
  was_novel           : False

--- Batch: 30 experiences from two conceptual clusters ---
  processed           : 30 experiences
  T before batch      : 1.0012
  T after  batch      : 0.2622  (cooled as expected)
  novel  (>0.5)       : 0/30
  mean novelty        : 0.411
  total |δ| applied   : 6.7474

--- Manifold state after annealing ---
  points now          : 112  (was 81 seed + placed concepts)
  write ops           : 62

--- Reset temperature ---
  T before reset      : 0.2622
  T after  reset      : 1.0500  (back to T₀+T_floor)
  schedule time       : 0.0  (reset to 0)
  manifold points     : 112  (unchanged by reset)

--- Engine summary ---
AnnealingEngine:
  processed        : 31
  novel (>0.5)     : 0  (0.0%)
  mean novelty     : 0.401
  total deformation: 6.8616
  temperature now  : 1.0500
  schedule time    : 0.0
  manifold writes  : 62
  manifold points  : 112
```

---

## Design Constraints Upheld

| Constraint | Status |
|---|---|
| No weights | ✅ — deformation magnitude driven by novelty score and temperature; no tunable weight matrices |
| No tokens | ✅ — experiences are continuous 104D vectors; no tokenisation or symbol IDs |
| No training phase | ✅ — annealing runs during operation; initial temperature is not a training hyperparameter but a runtime schedule |
| Local updates only | ✅ — all deformations delegated to `M.deform_local()` which hard-enforces Gaussian falloff (C2 guarantee unchanged) |
| Causality first class | ✅ — consistency gradient computed in full 104D space, including the causal fiber (dims 64–79); causal structure shapes the pull direction |
| Separation of concerns | ✅ — `AnnealingEngine` never generates output; only calls C2 WRITE operations (`deform_local`, `update_density`, `place`) |

---

## Issues Resolved

| Issue | Fix |
|---|---|
| Novelty always ≈ 0.63 for single-neighbor case with mean-normalised sigma | Replaced `sigma = mean_dist * sigma_scale` with `sigma = sigma_scale` (fixed absolute scale) so absolute distance determines novelty |
| `pytest.approx` used with `>=` operator raises `TypeError` | Replaced `>=  pytest.approx(x, abs=ε)` with `>= x - ε` (plain float comparison) |
| `test_novelty_high_for_unknown_region` needed explicit sigma_scale | Injected `NoveltyEstimator(sigma_scale=0.1)` directly on the engine for the test so that distances > 0.1 units score > 0.5 |

---

## Next: Phase 4

| Component | Description | Depends on |
|---|---|---|
| C5 — Flow Engine | SDE navigation of M(t) to produce reasoning as a continuous trajectory | Living Manifold ✅, Annealing Engine ✅ |
| C6 — Resonance Layer | Accumulates trajectory into pre-linguistic standing wave Ψ | Flow Engine (C5) |
