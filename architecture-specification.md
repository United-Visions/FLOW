# Geometric Causal Architecture — Full System Specification

**Version:** 0.1 — Foundational Spec  
**Status:** Pre-implementation  
**Purpose:** Define component boundaries, responsibilities, interfaces, and mathematical foundations for a weight-free, token-free reasoning architecture

---

## 0. Design Principles

Every decision in this specification follows from six constraints. Any proposed change must be evaluated against all six.

```
1. NO WEIGHTS
   Knowledge is never stored as tunable numerical parameters.
   Learning is geometric deformation, not gradient descent.

2. NO TOKENS
   Meaning is never discretized into symbol sequences.
   Output is a continuous trajectory rendered into language.

3. NO TRAINING PHASE
   There is no offline/online distinction.
   Growth is the operating mode, not a separate phase.

4. LOCAL UPDATES ONLY
   New knowledge deforms nearby geometry.
   Distant geometry is never affected by a local event.

5. CAUSALITY IS FIRST CLASS
   Cause-effect relationships are encoded structurally.
   Correlation and causation are architecturally distinct.

6. SEPARATION OF CONCERNS
   Meaning generation and language expression are separate systems.
   No component does more than one thing.
```

---

## 1. System Overview

The architecture consists of seven components arranged in two logical groups.

### Group A — The Manifold (What The System Knows)

```
┌─────────────────────────────────────────────────────────────┐
│  COMPONENT 1: SEED GEOMETRY ENGINE                           │
│  Computes the initial mathematical skeleton                  │
│  Runs once. Output is static forever.                        │
├─────────────────────────────────────────────────────────────┤
│  COMPONENT 2: LIVING MANIFOLD                                │
│  The dynamic high-dimensional geometric space                │
│  All knowledge lives here as shape                           │
├─────────────────────────────────────────────────────────────┤
│  COMPONENT 3: ANNEALING ENGINE                               │
│  Shapes the manifold from raw experience                     │
│  Runs continuously, never stops                              │
├─────────────────────────────────────────────────────────────┤
│  COMPONENT 4: CONTRAST ENGINE                                │
│  Places concepts via same/different relational judgments     │
│  Runs continuously alongside annealing                       │
└─────────────────────────────────────────────────────────────┘
```

### Group B — The Reasoner (What The System Does)

```
┌─────────────────────────────────────────────────────────────┐
│  COMPONENT 5: FLOW ENGINE                                    │
│  Navigates the manifold to produce meaning as trajectory     │
│  The reasoning process                                       │
├─────────────────────────────────────────────────────────────┤
│  COMPONENT 6: RESONANCE LAYER                                │
│  Accumulates holistic pre-linguistic meaning from flow       │
│  Produces the standing wave that gets rendered               │
├─────────────────────────────────────────────────────────────┤
│  COMPONENT 7: EXPRESSION RENDERER                            │
│  Converts resonance pattern to fluent language               │
│  Language-agnostic by construction                           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Experience in
     │
     ▼
[Seed Geometry Engine] ──once──▶ [Living Manifold] ◀── [Annealing Engine] ◀── raw experience
                                        │                [Contrast Engine] ◀── paired experience
                                        │
                                   Query in
                                        │
                                        ▼
                                 [Flow Engine]
                                        │
                                        ▼
                               [Resonance Layer]
                                        │
                                        ▼
                             [Expression Renderer]
                                        │
                                        ▼
                                  Language out
```

---

## 2. Component Specifications

---

### Component 1 — Seed Geometry Engine

**Responsibility:** Compute the mathematical skeleton of the manifold from first principles. This component runs exactly once and produces a static output that never changes.

**What it is not responsible for:** Placing any specific concepts. Responding to experience. Updating anything after initial computation.

#### Inputs

None. This component takes no external data. It derives geometry from mathematical structures alone.

#### Process

Compute four base geometries and compose them into a unified seed manifold.

**Base geometry 1 — Causal Structure**
```
Source:   The definition of causation (Pearl's do-calculus)
Geometry: Directed acyclic graph embedded in continuous space
Properties:
  - Directionality: causes precede effects as curvature
  - Asymmetry: causal edges are not reversible
  - Transitivity: causal chains compose geometrically
  - Interventional structure: do(X) vs observe(X) are spatially distinct
Output:   A directed Riemannian manifold with built-in time-like dimension
```

**Base geometry 2 — Logical Structure**
```
Source:   Propositional logic (Boolean algebra)
Geometry: Hypercube embedded in the manifold
Properties:
  - Contradiction: maximum distance (opposite vertices)
  - Entailment: directional proximity
  - Conjunction/disjunction: geometric intersection/union
  - Negation: reflection across a hyperplane
Output:   A logical topology — how propositions relate spatially
```

**Base geometry 3 — Probabilistic Structure**
```
Source:   Kolmogorov axioms + information geometry
Geometry: Statistical simplex with Fisher metric
Properties:
  - Certainty: vertices of the simplex
  - Uncertainty: interior of the simplex
  - Probability flow: natural geodesics along gradients
  - KL divergence: the natural distance metric
Output:   A confidence topology — uncertainty encoded as position
```

**Base geometry 4 — Similarity Structure**
```
Source:   The definition of similarity (metric space axioms)
Geometry: Metric space with flexible local curvature
Properties:
  - Distance = degree of difference
  - Triangle inequality enforced
  - Local curvature varies by domain density
  - Topology from metric alone — no coordinates required
Output:   A relational topology — closeness encoded as distance
```

**Composition**
```
The four base geometries are composed via fiber bundle construction:
- Similarity metric provides the base manifold (the floor)
- Causal structure provides directionality (the flow)
- Logical structure provides constraint topology (the walls)
- Probabilistic structure provides confidence gradients (the light)

Result: A single unified seed manifold M₀ with:
  - Riemannian metric from similarity structure
  - Causal fiber from causal DAG embedding
  - Logical constraint layer from hypercube topology
  - Probability density from statistical simplex
```

#### Output

`M₀` — the seed manifold. A fixed Riemannian manifold encoding the mathematical structure of causality, logic, probability, and similarity without a single data point.

#### Interfaces

```
Output → Component 2 (Living Manifold) as initial state
```

#### Mathematical Foundations

- Riemannian geometry (do Carmo)
- Directed acyclic graph embedding (topological sort → geometric embedding)
- Information geometry (Amari — Fisher metric, statistical manifolds)
- Algebraic topology (hypercube as CW complex)
- Fiber bundle theory (Steenrod — for composition)

---

### Component 2 — Living Manifold

**Responsibility:** Be the geometric space in which all knowledge lives. Receive deformations from the Annealing Engine and Contrast Engine. Provide position queries, distance queries, geodesic queries, and curvature queries to the Flow Engine and Resonance Layer.

**What it is not responsible for:** Deciding how to deform. Deciding what anything means. Generating output.

#### State

The manifold at any time t is characterized by:

```
M(t) = (M₀, φ(t), ρ(t), κ(t))

Where:
M₀   = seed geometry (fixed, from Component 1)
φ(t) = deformation field (how M₀ has been modified by experience)
ρ(t) = density function (how dense/sparse each region is)
κ(t) = curvature tensor (local curvature at every point)
```

#### Regions

The manifold contains three types of regions at any time:

```
CRYSTALLIZED REGIONS
  Definition:  High density, low flexibility
  Meaning:     Well-understood, frequently experienced concepts
  Properties:  Stiff geometry, resistant to further deformation
               Geodesics are stable and well-defined
               High confidence (probability mass concentrated)

FLEXIBLE REGIONS
  Definition:  Medium density, medium flexibility
  Meaning:     Partially understood, actively developing concepts
  Properties:  Geometry still deforming
               Geodesics less stable
               Moderate confidence

UNKNOWN TERRITORY
  Definition:  Low density, high flexibility
  Meaning:     Unexplored or poorly understood concepts
  Properties:  Sparse geometry, highly deformable
               Geodesics poorly defined
               Low confidence (probability mass diffuse)
```

#### Operations

```
READ operations (called by Flow Engine and Resonance Layer):
  position(concept)      → point P in M(t)
  distance(P₁, P₂)      → geodesic distance along M(t)
  geodesic(P₁, P₂)      → shortest path along M(t) surface
  curvature(P)           → curvature tensor at point P
  density(P)             → local density at point P
  neighbors(P, r)        → all points within radius r of P
  causal_direction(P₁, P₂) → causal flow vector from P₁ toward P₂

WRITE operations (called by Annealing Engine and Contrast Engine):
  deform_local(P, δ)     → apply deformation δ at point P
                           affect only points within locality radius
                           locality radius = f(density) — denser = smaller radius
  place(concept, P)      → create new point at position P
  update_density(P)      → recompute local density after placement
```

#### Locality Guarantee

This is a hard constraint. Any write operation must satisfy:

```
For any deformation at point P:
  effect(Q) → 0 as distance(P, Q) → ∞

No global updates. Ever.
New knowledge cannot reach across the manifold.
```

#### Interfaces

```
Input  ← Component 1 (initial state M₀)
Input  ← Component 3 (deformation operations)
Input  ← Component 4 (placement operations)
Output → Component 5 (all read operations)
Output → Component 6 (all read operations)
```

#### Mathematical Foundations

- Riemannian manifold theory (Riemannian metric, geodesics, curvature)
- Diffeomorphic deformation (smooth, bijective maps)
- Persistent homology (tracking topological changes over time)
- Dynamic metric spaces (evolving distance structures)

---

### Component 3 — Annealing Engine

**Responsibility:** Shape the manifold from raw unlabeled experience using physics-inspired self-organization. No labels. No supervision. No gradients.

**What it is not responsible for:** Creating the seed geometry. Placing specific named concepts. Generating output.

#### Process

```
TEMPERATURE SCHEDULE
  T(t) = T₀ · e^(-λt) + T_floor

  T₀      = initial temperature (high flexibility)
  λ        = cooling rate (tunable)
  T_floor  = minimum temperature (preserves some flexibility always)

  Early:  High T → geometry highly flexible, exploratory
  Later:  Low T  → geometry stiffens where confident
  Always: T_floor → unknown territory stays flexible
```

#### Experience Processing Loop

```
For each incoming raw experience E:

  1. LOCATE
     Find the natural position P for E in M(t) via resonance
     (where does E fit best given existing geometry?)

  2. MEASURE NOVELTY
     novelty(E) = f(distance to nearest neighbors, local density)
     High novelty → large deformation
     Low novelty  → small deformation (E is well-covered already)

  3. COMPUTE DEFORMATION
     δ(P) = novelty(E) · T(t) · gradient_toward_consistency(E, M(t))
     
     gradient_toward_consistency pulls:
       - Similar experiences toward each other
       - Dissimilar experiences apart
       - Causal relationships into directional alignment

  4. APPLY DEFORMATION (via Component 2 write)
     M.deform_local(P, δ)
     Locality radius = f(T(t)) — higher T means wider radius
     Lower T means deformation is increasingly local

  5. UPDATE DENSITY
     M.update_density(P)
```

#### Key Properties

```
Self-organizing:     No labels tell it where things should go
                     Physics of co-occurrence determines placement

Confidence encoding: Dense regions stiffened by repeated experience
                     Cannot be overwritten without enormous force
                     Sparse regions remain fluid and open

Temperature decay:   Early experiences shape coarse structure
                     Later experiences only fine-tune locally
                     T_floor ensures the system never fully freezes
```

#### Interfaces

```
Input  ← raw experience stream (continuous)
Input  ← Component 2 (read operations for resonance placement)
Output → Component 2 (deform_local, update_density)
```

---

### Component 4 — Contrast Engine

**Responsibility:** Place concepts precisely using only same/different relational judgments. Operates as persistent homology over pairwise relational data. Runs alongside the Annealing Engine, handling what raw exposure cannot precisely locate.

**What it is not responsible for:** Labeling. Classifying. Assigning meaning. Generating output.

#### The Core Operation

```
INPUT: A pair of experiences (E₁, E₂) and a judgment J ∈ {same, different}

PROCESS:
  current_dist = M.distance(position(E₁), position(E₂))

  If J = same:
    target_dist = current_dist · (1 - α)    [pull closer]
    
  If J = different:
    target_dist = current_dist · (1 + β)    [push apart]
  
  Compute displacement vectors for both points
  Apply via M.deform_local for each point
  Locality radius constrained to not affect third-party points

Where:
  α = attraction coefficient (tunable, ~0.1)
  β = repulsion coefficient (tunable, ~0.1)
```

#### Where Judgments Come From

```
SELF-SUPERVISED PAIRS
  Augmentations of the same experience → same
  Randomly sampled unrelated pairs    → different
  Temporal proximity                   → same (nearby in time)
  Causal proximity                     → same (cause-effect chains)

HUMAN FEEDBACK (optional, not required)
  Explicit corrections → same/different judgments
  Used as strong signal, not prerequisite
```

#### Persistent Homology Accumulation

```
The Contrast Engine maintains a persistence diagram:
  All pairwise distance relationships tracked over time
  Topological features (connected components, loops, voids) tracked
  Features that persist = real structure
  Features that die quickly = noise

Periodically fed back to Component 2 as structural corrections:
  "This cluster should be more connected"
  "This region should have a clear boundary"
  All expressed as deformations, not labels
```

#### Interfaces

```
Input  ← paired experience stream with same/different labels
Input  ← Component 2 (distance queries)
Output → Component 2 (deform_local for both points in pair)
```

---

### Component 5 — Flow Engine

**Responsibility:** Navigate the manifold in response to a query to produce meaning as a continuous trajectory. This is the reasoning process.

**What it is not responsible for:** Storing anything. Creating language. Understanding what the trajectory means symbolically.

#### Query Ingestion

```
A query Q enters as:
1. A perturbation injected into the manifold
   (Q is located in M via resonance — where does it fit?)
2. The starting position P₀ = M.position(Q)
3. Initial velocity V₀ = M.causal_direction(P₀, response_attractor)
   (response_attractor = the region of the manifold
    most relevant to answering Q)
```

#### The Flow Equation

```
The flow is governed by a stochastic differential equation
on the Riemannian manifold M(t):

dP = μ(P,t)dt + σ(P,t)dW

Where:
  P     = current position (point on M)
  μ     = drift vector (deterministic meaning pull)
  σ     = diffusion tensor (uncertainty and creative range)
  dW    = Riemannian Brownian motion
  t     = continuous time

The drift vector μ is the sum of four forces:
```

#### Force 1 — Semantic Gravity

```
F_gravity(P) = Σᵢ mᵢ · (Pᵢ - P) / distance(P, Pᵢ)²

Where:
  Pᵢ = nearby concept positions
  mᵢ = density at Pᵢ (denser = more massive = stronger pull)

Effect: Dense concept clusters attract the flow toward them
        Related ideas naturally draw the trajectory
        No explicit topic selection needed
```

#### Force 2 — Causal Curvature

```
F_causal(P) = κ_causal(P) · V_current

Where:
  κ_causal(P) = causal curvature tensor at P
  V_current   = current velocity vector

Effect: The manifold's causal structure bends the trajectory
        Flow naturally follows cause → effect chains
        Reasoning respects causal order by geometry
```

#### Force 3 — Contextual Momentum

```
F_momentum(P) = γ · V_previous

Where:
  V_previous = velocity vector from previous step
  γ          = momentum coefficient (~0.85)

Effect: Meaning has inertia — themes persist
        Sharp direction change = parameter drop = genuine topic shift
        Continuity of thought preserved geometrically
```

#### Force 4 — Contrast Repulsion

```
F_repulsion(P) = -Σⱼ contradiction_strength(P, Pⱼ) · (Pⱼ - P)

Where:
  Pⱼ = positions of concepts contradictory to current position
  contradiction_strength = logical incompatibility score from seed geometry

Effect: Contradictory regions repel the flow
        Logical coherence emerges from the geometry
        The flow avoids incoherence without instruction
```

#### Termination

```
Flow terminates when:
  - Velocity magnitude drops below threshold (natural conclusion)
  - Flow re-enters starting region (circular reasoning detected)
  - Maximum time budget reached (hard limit)
  - Flow reaches a stable attractor basin (answer crystallized)
```

#### Output

```
T = {(P₀, t₀), (P₁, t₁), ..., (Pₙ, tₙ)}

A time-stamped sequence of positions on M
representing the continuous trajectory of meaning
This is handed to Component 6 (Resonance Layer)
```

#### Interfaces

```
Input  ← query (as manifold perturbation)
Input  ← Component 2 (position, distance, geodesic, curvature, density queries)
Output → Component 6 (trajectory T)
```

---

### Component 6 — Resonance Layer

**Responsibility:** Convert the flow trajectory into a holistic pre-linguistic meaning representation — a standing wave across the manifold. The entire response exists here before rendering begins.

**What it is not responsible for:** Navigation. Language production. Storing knowledge.

#### How Resonance Accumulates

```
As the flow trajectory T passes through position P at time t:

1. EXCITATION
   All points Q within resonance radius r of P are excited:
   excitation(Q, t) = A · e^(-distance(P,Q)² / 2r²)
   
   A = amplitude (function of flow velocity — faster flow = stronger excitation)
   r = resonance radius (function of local curvature — curved regions resonate narrowly)

2. HARMONIC AMPLIFICATION
   Two regions harmonically related if their characteristic frequencies
   are integer multiples of each other:
   
   harmonic(P, Q) = true if freq(P) / freq(Q) ∈ ℤ⁺
   
   Harmonically related regions amplify each other's excitation
   Non-harmonic regions experience destructive interference

3. STANDING WAVE FORMATION
   Over the course of the trajectory, excitations accumulate:
   
   Ψ(Q) = ∫₀ᵀ excitation(Q, t) · harmonic_factor(Q, P(t)) dt
   
   Ψ(Q) = the standing wave amplitude at point Q
   
   High Ψ = strongly resonant with the trajectory = central to meaning
   Low Ψ  = weakly resonant = peripheral or irrelevant
   Zero Ψ = not part of the meaning at all
```

#### The Standing Wave As Complete Meaning

```
Ψ : M → ℝ⁺

A scalar field over the entire manifold.
Every point in M has an amplitude.
The pattern of amplitudes IS the complete pre-linguistic meaning.

High amplitude regions:  Core concepts of the response
Medium amplitude regions: Supporting context and nuance
Low amplitude regions:    Peripheral associations
Zero amplitude regions:   Not part of this response

The standing wave captures:
  - What the response is about (high amplitude positions)
  - How confident it is (density of high amplitude regions)
  - What is uncertain (sparse high amplitude in flexible regions)
  - What causes what (amplitude gradient following causal curvature)
  - What is peripheral (low amplitude)
  - What is irrelevant (zero amplitude)
```

#### Output

```
Ψ — the standing wave field over M
Handed to Component 7 (Expression Renderer)
```

#### Interfaces

```
Input  ← Component 5 (trajectory T)
Input  ← Component 2 (curvature, density, neighbor queries)
Output → Component 7 (standing wave Ψ)
```

---

### Component 7 — Expression Renderer

**Responsibility:** Convert the standing wave Ψ into fluent natural language. This component has no access to the manifold directly. It receives only Ψ and produces language. It is the only component that knows anything about language.

**What it is not responsible for:** Reasoning. Navigation. Knowledge storage. Any component above it.

#### The Rendering Pipeline

**Stage 1 — Segmentation**

```
INPUT: Standing wave Ψ

Identify natural segments from the wave structure:
  - A segment boundary occurs where Ψ has a local minimum
    (the flow passed between two distinct regions)
  - Not fixed length. Not token-aligned.
  - Segments emerge from the meaning structure itself.

OUTPUT: Ordered sequence of wave segments Ψ₁, Ψ₂, ..., Ψₖ
        Each segment = a coherent chunk of meaning
        Order = trajectory order (temporal from the flow)
```

**Stage 2 — Resonance Matching**

```
INPUT: Each segment Ψᵢ

For each segment, find the linguistic expression E that minimizes:
  resonance_distance(Ψᵢ, semantic_wave(E))

Where:
  semantic_wave(E) = the resonance profile of expression E
                     computed by running E through a lightweight
                     forward model of the manifold

This is constraint satisfaction, not token prediction:
  - Many candidate expressions evaluated simultaneously
  - The one whose meaning-wave best matches Ψᵢ is selected
  - Length, register, and complexity emerge from Ψᵢ's structure

OUTPUT: Best-matching expression Eᵢ for each segment Ψᵢ
```

**Stage 3 — Flow Preservation**

```
INPUT: Ordered expressions E₁, E₂, ..., Eₖ

Ensure the linguistic rendering preserves the trajectory's dynamics:
  - Fast flow → short, direct sentences
  - Slow rich flow → long complex sentences
  - Trajectory loop (reinforcement) → anaphora, repetition
  - Sharp direction change → paragraph break, transitional phrase
  - High uncertainty region → hedged language, conditionals
  - Crystallized region → declarative, confident language

Adjust expression boundaries and connectives to preserve momentum.

OUTPUT: Final language output
```

#### Language Agnosticism

```
At no point does the renderer know which language it is rendering into.
The same Ψ can be handed to:

  English renderer      → English output
  French renderer       → French output
  Mathematical renderer → Formal notation output
  Musical renderer      → Score or MIDI output
  Visual renderer       → Diagram or animation spec

Each renderer learns the mapping from wave structure to its medium.
The model's reasoning (Components 1–6) is entirely untouched.
```

#### Interfaces

```
Input  ← Component 6 (standing wave Ψ)
Input  ← target language specification
Output → fluent language in target language
```

---

## 3. Cross-Cutting Concerns

### Causality Encoding

Causality is not handled by a single component. It is woven through the architecture:

```
Seed Geometry Engine:    Encodes causal DAG structure as directed curvature
Living Manifold:         Exposes causal_direction() as a first-class query
Annealing Engine:        Pulls co-occurring experiences into causal alignment
Flow Engine:             Force 2 (Causal Curvature) follows causal direction
Resonance Layer:         Amplitude gradients respect causal flow direction
Expression Renderer:     Causal chains emerge as natural sentence ordering
```

### Uncertainty Representation

```
Seed Geometry Engine:    Encodes probability simplex as uncertainty geometry
Living Manifold:         Flexible regions = uncertain territory (low density)
Annealing Engine:        Uncertainty = geometric flexibility, not numerical probability
Flow Engine:             Diffusion term σ(P,t) is larger in flexible regions
Resonance Layer:         Low amplitude in sparse regions = uncertain meaning
Expression Renderer:     Sparse high-amplitude regions → hedged language
```

### Memory

```
There is no separate memory component.
The manifold IS the memory.

Working memory:  The current trajectory (Component 5 state)
Long term memory: The crystallized regions of the manifold
Episodic memory:  The trajectory history (a path through M over time)
Forgetting:       Does not occur — old geometry does not move
Interference:     Does not occur — updates are local only
```

---

## 4. Component Dependency Map

```
                [1 Seed Geometry Engine]
                          │
                          │ (one-time initialization)
                          ▼
                  [2 Living Manifold]
                 ↗         ↑
[3 Annealing Engine]    [4 Contrast Engine]
(continuous writes)     (continuous writes)

                  [2 Living Manifold]
                 ↘         ↘
          [5 Flow Engine] ──────▶ [6 Resonance Layer]
          (reads only)    (reads)           │
                                            │
                                            ▼
                                  [7 Expression Renderer]
                                  (no manifold access)
```

**Strict rules:**
- Component 7 has no access to Components 1–5. Only the standing wave Ψ.
- Components 3 and 4 have write access to Component 2 only.
- Components 5 and 6 have read access to Component 2 only.
- Component 1 writes to Component 2 once and never again.

---

## 5. What Needs To Be Built

### Immediately Tractable

```
COMPONENT 1 — SEED GEOMETRY ENGINE
  The mathematics exists (Riemannian geometry, info geometry, DAG embedding)
  Algorithmic challenge: composing four base geometries into one manifold
  Estimated difficulty: High (mathematics), Medium (implementation)
  Blocking: Nothing. Can start now.
```

### Research Required

```
COMPONENT 2 — LIVING MANIFOLD (data structures)
  Dynamic Riemannian manifold at scale does not exist as a data structure
  Incremental geodesic computation is an open problem
  Local deformation with guaranteed locality is partially solved
  Estimated difficulty: Extreme
  Blocking: Components 3, 4, 5, 6

COMPONENT 5 — FLOW ENGINE (at scale)
  SDE on Riemannian manifolds: mathematically solved
  Numerical SDE solvers for high-dimensional curved spaces: research problem
  Real-time performance: major engineering challenge
  Estimated difficulty: Very High
  Blocking: Component 6

COMPONENT 6 — RESONANCE LAYER
  Wave mechanics on manifolds: mathematically understood
  Efficient standing wave accumulation at scale: does not exist
  Harmonic structure of concept space: unknown empirically
  Estimated difficulty: Very High
  Blocking: Component 7

COMPONENT 7 — EXPRESSION RENDERER
  Resonance-to-language mapping: no prior art
  Constraint satisfaction for expression: closest is program synthesis
  Flow preservation in prose: no prior art
  Estimated difficulty: Very High
  Blocking: Nothing (can be prototyped with mock wave input)
```

### Hardware Problem

```
All current hardware is optimized for matrix multiplication.
This architecture requires:
  - Efficient geodesic computation (graph algorithms, not matmul)
  - Local manifold deformation (spatial data structures)
  - Wave accumulation (signal processing, not matrix ops)
  - High-dimensional nearest-neighbor search (approximate, real-time)

Closest existing candidates:
  - Neuromorphic hardware (Intel Loihi, IBM NorthPole)
  - Optical computing (wave mechanics native)
  - Spatial computing hardware (graph processors)

None are sufficient today. Custom hardware is the long-term requirement.
```

---

## 6. Build Order

```
PHASE 1 — FOUNDATIONS (can start now)
  1a. Implement Seed Geometry Engine
      Derive and compose the four base geometries
      Validate that the resulting manifold has correct properties
      
  1b. Prototype Expression Renderer
      Start with mock wave input (hand-crafted Ψ)
      Build resonance-matching as constraint satisfaction
      Validate that it produces coherent language from wave structure

PHASE 2 — THE MANIFOLD (requires new data structures)
  2a. Design Living Manifold data structure
      Prototype at small scale (thousands of points)
      Validate locality guarantee mathematically
      
  2b. Implement Contrast Engine
      Simpler of the two shaping components
      Validates manifold write operations at small scale

PHASE 3 — SHAPING (requires Phase 2)
  3a. Implement Annealing Engine
      Validate that geometry self-organizes from raw exposure
      Benchmark against known concept similarity datasets

PHASE 4 — REASONING (requires Phase 2 at scale)
  4a. Implement Flow Engine
      Validate that trajectories produce coherent meaning paths
      Benchmark geodesic computation performance
      
  4b. Implement Resonance Layer
      Validate standing wave formation from known trajectories
      Connect to Expression Renderer

PHASE 5 — INTEGRATION
  5a. Full pipeline end-to-end
  5b. Develop new evaluation framework
      (existing benchmarks are inappropriate — all assume tokens and weights)
```

---

## 7. What Success Looks Like

```
MINIMUM VIABLE DEMONSTRATION
  A system that can:
  - Accept a natural language query
  - Navigate a small (10,000 point) hand-crafted manifold
  - Produce a standing wave
  - Render that wave into coherent language
  - WITHOUT any weight matrices
  - WITHOUT any tokenization
  - WITHOUT any training phase
  
  This demonstrates the architecture is sound.
  Scale is a separate problem.

FULL SUCCESS CRITERIA
  One-shot learning:     New concept placed correctly from one example
  No catastrophic forgetting: Old knowledge unaffected after 10,000 new concepts
  Causal reasoning:      Correct intervention vs observation distinction
  Interpretability:      Trajectory through manifold explains any answer
  Language agnosticism:  Same Ψ renders correctly into two different languages
```
