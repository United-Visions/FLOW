"""Phase 5 end-to-end demo — Full Integration Pipeline & Evaluation Framework.

Exercises the complete Geometric Causal Architecture C1 → C7 via
GEOPipeline and PipelineEvaluator.
"""
import numpy as np

from src.phase3.annealing_engine.experience import Experience
from src.phase5.pipeline.pipeline import GEOPipeline
from src.phase5.evaluation.evaluator import PipelineEvaluator

print("=== FLOW — Phase 5 Demo ===\n")

# ── Build the full pipeline (C1 → C7) ────────────────────────────────────────
pipeline = GEOPipeline(T0=0.8, lambda_=0.02, T_floor=0.02, flow_seed=42)
print(pipeline.summary())

# ── Enrich the manifold with experience via C3 Annealing Engine ───────────────
print("\n--- Annealing (C3): learning from raw experience ---")
rng = np.random.default_rng(42)
M = pipeline.manifold
base_causal = M.position("causal::perturbation").copy()
base_math   = M.position("domain::mathematical").copy()
base_mech   = M.position("causal::mechanism").copy()

experiences = (
    [Experience(base_causal + rng.standard_normal(104) * 0.05,
                label=f"demo::causal_{i}") for i in range(6)]
    + [Experience(base_math + rng.standard_normal(104) * 0.05,
                  label=f"demo::math_{i}")   for i in range(6)]
    + [Experience(base_mech + rng.standard_normal(104) * 0.05,
                  label=f"demo::mech_{i}")   for i in range(4)]
)
results = pipeline.learn_batch(experiences)
print(f"  Experiences processed : {len(results)}")
print(f"  Mean novelty          : {sum(r.novelty for r in results) / len(results):.4f}")
print(f"  Temperature T(t)      : {pipeline.temperature:.4f}")
print(f"  Concepts on M(t)      : {pipeline.n_concepts}")

# ── Shape the manifold with contrast judgments via C4 ─────────────────────────
print("\n--- Contrast (C4): same/different relational judgments ---")
cr_same = pipeline.contrast("demo::causal_0", "demo::causal_1", "same")
print(
    f"  SAME  causal_0 / causal_1  "
    f"dist_before={cr_same.distance_before:.4f}  "
    f"dist_after={cr_same.distance_after:.4f}  "
    f"Δ={cr_same.distance_change:+.4f}"
)
cr_diff = pipeline.contrast("demo::causal_0", "demo::math_0", "different")
print(
    f"  DIFF  causal_0 / math_0    "
    f"dist_before={cr_diff.distance_before:.4f}  "
    f"dist_after={cr_diff.distance_after:.4f}  "
    f"Δ={cr_diff.distance_change:+.4f}"
)

# ── Full pipeline queries: C5 → C6 → C7 ──────────────────────────────────────
print("\n--- Full pipeline queries (C5 → C6 → C7) ---")
queries = [
    (base_causal, "what causes perturbation?"),
    (base_math,   "what is the mathematical structure?"),
    (base_mech,   "describe the mechanism"),
]
for vec, label in queries:
    result = pipeline.query(vec, label=label)
    print(
        f"  query  : {label!r}\n"
        f"  steps  : {result.n_steps}  reason={result.termination_reason}\n"
        f"  wave Ψ : {len(result.wave.points)} points, "
        f"confidence={result.wave_confidence:.3f}\n"
        f"  render : confidence={result.confidence:.3f}  "
        f"flow_preserved={result.flow_preserved}\n"
        f"  text   : {result.text[:120]!r}{'...' if len(result.text)>120 else ''}\n"
    )

print(f"  Total queries issued  : {pipeline.query_count}")

# ── Evaluation framework ──────────────────────────────────────────────────────
print("\n--- Evaluation Framework ---")
evaluator = PipelineEvaluator(pipeline)

# 1. Single-query coherence
ev_q = evaluator.evaluate_query(base_causal, label="eval::causal")
print("  [Coherence — causal query]")
print(f"    overall score       : {ev_q.overall_score():.4f}")
print(f"    wave confidence     : {ev_q.coherence.wave_confidence:.4f}")
print(f"    render confidence   : {ev_q.coherence.render_confidence:.4f}")
print(f"    core fraction       : {ev_q.coherence.core_fraction:.4f}")
print(f"    trajectory steps    : {ev_q.coherence.trajectory_steps}")
print(f"    termination reason  : {ev_q.coherence.termination_reason}")

# 2. Causal direction metric
print("\n  [Causal direction]")
causal_m = evaluator.evaluate_causal_direction(
    cause_vec=base_causal, effect_vec=base_mech,
    cause_label="eval::cause_node", effect_label="eval::effect_node",
)
print(f"    causal_score        : {causal_m.causal_score:.4f}")
print(f"    forward steps       : {causal_m.forward_steps}")
print(f"    backward steps      : {causal_m.backward_steps}")
print(f"    forward mean speed  : {causal_m.forward_speed:.4f}")
print(f"    backward mean speed : {causal_m.backward_speed:.4f}")
print(f"    causal fiber norm   : {causal_m.causal_direction:.4f}  (1.0 = orthogonal fibers)")

# 3. Novelty decay
print("\n  [Novelty decay — repeated exposure]")
decay = evaluator.evaluate_novelty_decay(base_causal, label="eval::decay", n_reps=5)
print(f"    novelty scores      : {[round(v, 4) for v in decay]}")
print(f"    monotonically dec.  : {all(decay[i] >= decay[i+1] for i in range(len(decay)-1))}")

# 4. Locality check
print("\n  [Locality check]")
loc = evaluator.evaluate_locality(
    rng.standard_normal(104), label="eval::locality_anchor"
)
print(f"    locality satisfied  : {loc.locality_satisfied}")
print(f"    n_nearby_moved      : {loc.n_nearby_moved}")
print(f"    n_distant_moved     : {loc.n_distant_moved}")
print(f"    max_distant_shift   : {loc.max_distant_shift:.2e}")
print(f"    locality_radius     : {loc.locality_radius_used:.4f}")

# 5. Full evaluation suite
print("\n  [Full evaluation suite]")
suite = evaluator.run_suite(
    vectors=[base_causal, base_math, base_mech],
    labels=["suite::causal", "suite::math", "suite::mech"],
    novelty_reps=3,
)
d = suite.as_dict()
print(f"    n_queries           : {d['n_queries']}")
print(f"    mean_coherence      : {d['mean_coherence']:.4f}")
print(f"    mean_render_conf    : {d['mean_render_confidence']:.4f}")
print(f"    mean_wave_conf      : {d['mean_wave_confidence']:.4f}")
print(f"    mean_steps          : {d['mean_steps']:.1f}")
print(f"    terminations        : {d['termination_distribution']}")
print(f"    causal_score        : {d['causal_score']:.4f}")
print(f"    locality_satisfied  : {d['locality_satisfied']}")
print(f"    novelty_is_decaying : {d['novelty_is_decaying']}")
print(f"    novelty_decay       : {[round(v, 4) for v in suite.novelty_decay]}")

print("\n=== Phase 5 demo complete ===")
