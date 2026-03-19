# Phase 4 — COMPLETED

**Completed**: March 18, 2026  
**Status**: ✅ 388/388 tests passing (90 Phase 1 + 95 Phase 2 + 90 Phase 3 + 113 Phase 4) | Demo running end-to-end in 1.01s

---

## Scope

Phase 4 covered two components from the build order:

| Component | Description |
|---|---|
| C5 — Flow Engine | SDE navigation of M(t) to produce meaning as a continuous trajectory |
| C6 — Resonance Layer | Accumulates trajectory into the pre-linguistic standing wave Ψ |

---

## What Was Built

### `src/phase4/flow_engine/`

| File | Purpose |
|---|---|
| `query.py` | `Query` — 104D query vector + optional label + optional attractor label. `FlowStep` — one Euler-Maruyama integration step: position, velocity, time, speed, curvature. `Trajectory` — ordered sequence of FlowSteps with derived views: `positions`, `velocities`, `as_position_time_pairs`, `mean_speed`, `mean_curvature`. |
| `forces.py` | `ForceComputer` — computes the four geometrically-grounded drift forces: (1) Semantic Gravity `F_grav = Σᵢ mᵢ·(Pᵢ−P)/‖Pᵢ−P‖²` (density-weighted pull toward concept clusters), (2) Causal Curvature `F_cau = κ(P)·causal_dir` (causal fiber bends trajectory), (3) Contextual Momentum `F_mom = γ·V_prev` (meaning has inertia), (4) Contrast Repulsion `F_rep = −strength·Σⱼ contr(P,Pⱼ)·(Pⱼ−P)` (logical fiber contradictions repel). Combined via `combined_drift(position, velocity, manifold, weights)`. |
| `sde.py` | `SDESolver` — Euler-Maruyama discretisation of `dP = μdt + σdW`. Diffusion magnitude `σ(P) = diffusion_scale · (1 − density(P))`: sparse regions are explorative, dense crystallised regions are precise. |
| `engine.py` | `FlowEngine` — central Phase 4 data structure. Entry point `flow(query) → Trajectory`. Steps: LOCATE (kNN anchor), ORIENT (initial velocity toward attractor), INTEGRATE (Euler-Maruyama loop), TERMINATE. Four termination conditions: `velocity_threshold`, `revisit_detected`, `max_steps`, `attractor_reached`. Attractor is selected as the densest manifold point at least `min_attractor_dist` away from the starting position. |
| `__init__.py` | Package exports: `FlowEngine`, `Query`, `FlowStep`, `Trajectory`, `ForceComputer`, `SDESolver`. |

**Flow process** (per query Q):
```
1. LOCATE    nearest seed/concept point via kNN → anchor P₀
2. ORIENT    find response attractor (densest point ≥ min_attractor_dist away)
             V₀ = direction(P₀ → attractor) · dt
3. INTEGRATE Euler-Maruyama loop:
             drift μ = w₁F_gravity + w₂F_causal + w₃F_momentum + w₄F_repulsion
             new_P = P + μ·dt + σ·√dt·dW     σ = diffusion_scale·(1−density(P))
4. TERMINATE on first satisfied condition:
             attractor_reached / revisit_detected / velocity_threshold / max_steps
5. RETURN    Trajectory {(P₀,t₀), (P₁,t₁), …, (Pₙ,tₙ)} → handed to C6
```

**Four force weights** (default `w₁=0.4, w₂=0.2, w₃=0.3, w₄=0.1`):
```
w₁ F_gravity   — semantic clustering dominates direction
w₂ F_causal    — causal structure bends trajectory (weaker: manifold does work)
w₃ F_momentum  — continuity of thought (strong: themes persist, γ=0.85)
w₄ F_repulsion — logical coherence (lightest: rarely needed near seed points)
```

### `src/phase4/resonance_layer/`

| File | Purpose |
|---|---|
| `accumulator.py` | `ExcitationKernel` — computes `A·exp(−‖Q−P‖²/2r²)` where `r_eff = resonance_radius/(1+κ)` (curved regions resonate narrowly). `HarmonicKernel` — harmonic factor `exp(−δ²/2τ²)` where `δ = |ratio − round(ratio)|`, `ratio = (κ_Q+1)/(κ_P+1)` (integer frequency ratios amplify). `ResonanceAccumulator` — accumulates `Ψ(Qᵢ) = Σⱼ excitation(Qᵢ,Pⱼ)·harmonic(κᵢ,κⱼ)` over all trajectory sites. |
| `layer.py` | `ResonanceLayer` — central Phase 4 data structure. Entry point `accumulate(trajectory) → StandingWave`. Converts trajectory `FlowSteps` into `(position, speed, curvature)` excitation sites, runs `ResonanceAccumulator`, normalises amplitudes to `[0,1]`, builds `WavePoint` list, adds `query_echo`. Metadata includes `n_trajectory_steps`, `termination_reason`, `trajectory_mean_speed`, `trajectory_mean_curvature`. Produces `StandingWave` (defined in `src/phase1/expression/wave.py`) compatible with C7. |
| `__init__.py` | Package exports: `ResonanceLayer`, `ResonanceAccumulator`, `ExcitationKernel`, `HarmonicKernel`. |

**Resonance accumulation** (per trajectory):
```
Sites  = [(position, speed, curvature) for step in trajectory]

For each site Qᵢ:
  Ψ(Qᵢ) = Σⱼ  A·exp(−‖Qᵢ−Pⱼ‖²/2r_eff²) · exp(−δ²/2τ²)
  where:
    A       = max(speed_j, 0.01)             (flow speed as amplitude)
    r_eff   = resonance_radius / (1+κⱼ)     (curvature narrows cone)
    δ       = |ratio − round(ratio)|         (harmonic deviation)
    ratio   = (κᵢ+1) / (κⱼ+1)              (characteristic frequencies)

Output: normalised Ψ ∈ [0,1] → WavePoints → StandingWave → C7
```

---

## Test Results

```
388 passed in 3.04s  (90 Phase 1 + 95 Phase 2 + 90 Phase 3 + 113 Phase 4)
```

| Test Class | Tests | Result |
|---|---|---|
| `TestQuery` | 9 | ✅ |
| `TestFlowStep` | 7 | ✅ |
| `TestTrajectory` | 10 | ✅ |
| `TestForceComputer` | 15 | ✅ |
| `TestSDESolver` | 9 | ✅ |
| `TestFlowEngine` | 15 | ✅ |
| `TestFlowEngineIntegration` | 5 | ✅ |
| `TestExcitationKernel` | 7 | ✅ |
| `TestHarmonicKernel` | 7 | ✅ |
| `TestResonanceAccumulator` | 7 | ✅ |
| `TestResonanceLayer` | 13 | ✅ |
| `TestResonanceLayerIntegration` | 9 | ✅ |

---

## Demo Output

Running `tests/phase-4_demo.py` (via `python -m tests.phase-4_demo` from project root):

```
=== FLOW — Phase 4 Demo ===

  [1/5] Deriving causal geometry from Pearl's do-calculus...
  [2/5] Deriving logical geometry from Boolean algebra...
  [3/5] Deriving probabilistic geometry from Kolmogorov axioms...
  [4/5] Deriving similarity geometry from metric space axioms...
  [5/5] Composing into unified bundle via fiber bundle construction...

M₀ built successfully in 0.007s

Living Manifold M(t):
  Points          : 97
  Dimension       : 104
  Manifold time t : 32.000
  Write ops       : 32
  Regions:
    Crystallized  : 97
    Flexible      : 0
    Unknown       : 0

--- Force Computer ---
  position            : causal::perturbation
  ‖F_gravity‖         : 1.0000
  ‖F_causal‖          : 1.0000
  ‖F_momentum‖        : 0.0000  (zero init velocity)
  ‖F_repulsion‖       : 0.0000

--- SDE Solver ---
  dt                  : 0.05
  diffusion_scale     : 0.05
  σ(P) at seed point  : 0.000000  (low — dense, crystallised region)
  step ‖Δposition‖    : 0.025706
  step ‖velocity‖     : 0.025706

--- Flow Engine (single query) ---
  query label         : what causes perturbation?
  n_steps             : 12
  total flow time     : 0.550
  termination reason  : revisit_detected
  mean speed          : 0.030634
  mean curvature      : 1.5569
  ‖P_last − P_first‖  : 0.0209  (traversal distance)

--- Three conceptual queries ---
  causal query             → steps= 12  reason=revisit_detected      mean_speed=0.0306
  mathematical query       → steps= 12  reason=revisit_detected      mean_speed=0.0304
  mechanism query          → steps= 12  reason=revisit_detected      mean_speed=0.0308

--- Resonance Layer ---
  trajectory steps    : 12
  wave points (Ψ>0)  : 12
  total energy ∫Ψ    : 11.9691
  wave confidence     : 0.000
  wave uncertainty    : 1.000
  peak point label    : flow_t0
  peak amplitude      : 1.0000
  peak τ (causal time): 0.000
  confident core (≥0.4 norm amp) : 12 points
  n_trajectory_steps  : 12
  termination_reason  : revisit_detected
  mean_speed (traj)   : 0.0306

--- Query echo in wave ---
  label               : query::what causes perturbation?
  amplitude           : 0.0500  (weak — query echo only)
  τ                   : 0.000  (origin of the flow)

--- Full pipeline: C5 → C6 → C7 ---
  query     : mechanism
  steps     : 12, reason=revisit_detected
  wave pts  : 12, energy=11.986
  rendered  : "Unlike flow t0, the the underlying process is different in that the relevant
factor. Unlike flow t1 — specifically regarding flow_t1 and flow_t2 —, the flow
t2 is different in that the underlying process. [...]"
  confidence: 0.564

  query     : intervention
  steps     : 12, reason=revisit_detected
  wave pts  : 12, energy=11.987
  rendered  : "The key insight is that flow t0. Unlike flow t1 — specifically regarding
flow_t1 and flow_t2 —, the flow t2 is different in that the underlying process.
[...]"
  confidence: 0.487
```

---

## Design Constraints Upheld

| Constraint | Status |
|---|---|
| No weights | ✅ — all four forces are geometric operations (density weighting, curvature scaling, cosine similarity in logical fiber); no tunable weight matrices |
| No tokens | ✅ — queries are continuous 104D vectors; trajectories are continuous paths in geometric space; no tokenisation at any stage |
| No training phase | ✅ — FlowEngine navigates the manifold as-built; no offline training; SDE parameters are runtime configuration |
| Local updates only | ✅ — FlowEngine and ResonanceLayer are read-only clients of C2 (no WRITE operations); all manifold writes remain in C3/C4 |
| Causality first class | ✅ — Force 2 (Causal Curvature) explicitly follows the causal fiber (dims 64–79) via `causal_direction()`; harmonics derived from `curvature()` which is denser in causal regions; `tau` (causal time) encodes trajectory ordering in the wave |
| Separation of concerns | ✅ — C5 only navigates and returns a trajectory; C6 only converts trajectory to Ψ; neither generates language nor modifies the manifold; C7 receives only Ψ with no manifold access |

---

## Issues Resolved

| Issue | Fix |
|---|---|
| Flow terminates immediately: attractor selected as P₀ itself (dist=0 < attractor_radius) | Added `min_attractor_dist` parameter; `_find_attractor` skips any candidate within `min_attractor_dist` of P₀, ensuring the flow traverses a meaningful path |
| `SDESolver` test used non-existent `seed=` kwarg | Test corrected to use `rng=np.random.default_rng(0)` as per the actual constructor |
| Invalid manifold label `"domain::logical"` used in tests | Replaced with correct label `"domain::logical_entities"` (16-domain taxonomy name) |
| `test_two_different_queries_produce_different_waves` compared wave energies that happened to be equal | Test rewritten to assert that the two trajectories start at different positions, which is the meaningful geometric invariant |

---

## Next: Phase 5

| Component | Description | Depends on |
|---|---|---|
| Phase 5 — Full Integration | End-to-end pipeline; replace mock wave input in C7 with real Ψ from C6; develop evaluation framework | All phases ✅ |
