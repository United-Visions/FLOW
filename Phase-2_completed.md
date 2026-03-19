# Phase 2 — COMPLETED

**Completed**: March 18, 2026  
**Status**: ✅ 185/185 tests passing (90 Phase 1 + 95 Phase 2) | Demo running end-to-end in 1.23s

---

## Scope

Phase 2 covered two components from the build order:

| Component | Description |
|---|---|
| C2 — Living Manifold | Dynamic Riemannian manifold M(t) wrapping M₀, with full READ + WRITE API |
| C4 — Contrast Engine | Same/different relational judgments that self-organise the geometry |

---

## What Was Built

### `src/phase2/living_manifold/`

| File | Purpose |
|---|---|
| `state.py` | `DeformationField` φ(t), `DensityField` ρ(t), `ManifoldState` — mutable state containers for M(t). Accumulates displacements per label, tracks write counter and manifold time. |
| `regions.py` | `RegionClassifier` — maps density scalar to `RegionType` (CRYSTALLIZED / FLEXIBLE / UNKNOWN). Derives stiffness, flexibility, locality radius (`r = r_max · exp(−ρ·3)`), diffusion scale. |
| `geodesic.py` | `GeodesicComputer` — maintains a kNN graph over all labelled points. Dijkstra-based geodesic path and distance. Incremental `add_point` / `update_point` without full rebuild. |
| `deformation.py` | `LocalDeformation` — Gaussian-weighted displacement kernel. Locality guarantee: effect → 0 as distance → ∞. Stiffness resists neighbourhood drag; centre point always displaced at full weight. |
| `manifold.py` | `LivingManifold` — central Phase 2 data structure. 81 seed points loaded from M₀. Lazy KD-tree for fast neighbour lookups. Full READ + WRITE interface (see below). |

**READ operations** (called by Flow Engine / Resonance Layer):
`position()`, `distance()`, `geodesic()`, `geodesic_distance()`, `curvature()`, `density()`, `neighbors()`, `nearest()`, `causal_direction()`, `domain_of()`, `locality_radius()`, `causal_ancestry()`, `confidence()`, `region_type()`, `logic_certainty()`

**WRITE operations** (called by Annealing Engine / Contrast Engine):
`place()` — add or move a concept to an exact position  
`deform_local()` — Gaussian-weighted local displacement, returns n_affected  
`update_density()` — recompute density for a label after external changes

### `src/phase2/contrast_engine/`

| File | Purpose |
|---|---|
| `persistence.py` | `PersistenceDiagram` — records pairwise distance observations over time. Detects birth/death events when pairs cross the cluster threshold. `get_persistent_features()`, `cluster_corrections()` — proposes tighten/separate corrections for long-lived pairs. |
| `engine.py` | `ContrastEngine` — applies SAME/DIFFERENT judgments to the Living Manifold. SAME → pull P₁ and P₂ closer by α. DIFFERENT → push apart by β. Returns `ContrastResult` with before/after distances and displacement vectors. Periodically applies persistence-based corrections. |

---

## Test Results

```
185 passed in 1.23s  (90 Phase 1 + 95 Phase 2)
```

| Test Class | Tests | Result |
|---|---|---|
| `TestDeformationField` | 6 | ✅ |
| `TestDensityField` | 4 | ✅ |
| `TestManifoldState` | 5 | ✅ |
| `TestRegionClassifier` | 8 | ✅ |
| `TestGeodesicComputer` | 7 | ✅ |
| `TestLocalDeformation` | 9 | ✅ |
| `TestLivingManifold` | 26 | ✅ |
| `TestPersistenceDiagram` | 9 | ✅ |
| `TestContrastEngine` | 21 | ✅ |

---

## Demo Output

Running `tests/phase-2_demo.py`:

```
Living Manifold M(t):
  Points       : 81
  Dimension    : 104
  Regions      : 81 crystallized, 0 flexible, 0 unknown

READ API:
  distance(perturbation → direct_effect) : 2.0414
  causal_ancestry(p → q)                 : True
  region_type(p)                         : crystallized
  locality_radius(p)                     : 0.2489
  confidence(p)                          : 1.000
  domain_of(p)                           : causal_mechanisms
  nearest(p, k=3) : [perturbation, initial_state, co_occurrence]
  geodesic path   : 3 waypoints

WRITE — place new concept:
  n_points after  : 82

WRITE — deform_local:
  points affected : 1
  centre shifted  : 0.050000
  write ops       : 2

Contrast Engine — 5 judgments:
  [same     ] perturbation  ↔ direct_effect     → Δdist -0.2786  (closer)
  [different] perturbation  ↔ epistemic         → Δdist +0.2758  (farther)
  [same     ] direct_effect ↔ downstream_effect → Δdist -0.1835  (closer)
  [different] mathematical  ↔ epistemic         → Δdist +0.2210  (farther)
  [same     ] perturbation  ↔ propagation       → Δdist -0.2500  (closer)

  correct direction : 100.0%
  tracked pairs     : 5
  persistent features (≥1 lifetime): 4
  cluster corrections proposed      : 4
```

---

## Design Constraints Upheld

| Constraint | Status |
|---|---|
| No weights | ✅ — geometry self-organises via relational judgments, no weight matrices |
| No training phase | ✅ — M(t) evolves from judgments only, no gradient descent |
| Local updates only | ✅ — `deform_local` hard-enforces Gaussian falloff; effect → 0 at distance |
| Causality first class | ✅ — causal fiber slice (dims 64–79) used directly for ancestry queries |
| Separation of concerns | ✅ — Contrast Engine never reads geometry directly; goes via `LivingManifold` API |

---

## Issues Resolved

| Issue | Fix |
|---|---|
| `FiberBundleComposer()` called with no args in `LivingManifold.__init__` | Replaced with `seed.composer` — reuses the already-initialised composer from M₀ |
| `causal_ancestry(p, q)` with `np.ndarray` crashed on `.causal_fiber` | Replaced delegation to `SeedManifold` with direct slice `p[64:80]` on raw vectors |
| `domain_of(p)` with `np.ndarray` crashed on `.base` | Replaced delegation with direct slice `p[0:64]` into `SimilarityGeometry.domain_of` |
| `logic_certainty(p)` with `np.ndarray` crashed on `.logical_fiber` | Replaced with `1 - log.uncertainty_score(p[80:88])` directly |
| `deform_local` returning 0 affected for high-density regions | Centre point now always displaced at full weight — crystallisation resists *neighbourhood drag*, not deliberate targeted movement |

---

## Next: Phase 3

| Component | Description | Depends on |
|---|---|---|
| C3 — Annealing Engine | Continuous self-organisation of M(t) via simulated annealing on the geometry | Living Manifold ✅, Contrast Engine ✅ |
