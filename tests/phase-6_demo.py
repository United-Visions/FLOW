"""Phase 6 demo — Coherent Language Output from Real Manifold Concepts.

Demonstrates the three Phase 6 improvements to C6 and C7:

  1. Semantic WavePoint labels   — C6 (ResonanceLayer) resolves each trajectory
                                   step to its nearest manifold concept label
                                   instead of a positional placeholder.

  2. Clean concept rendering     — C7 (ExpressionRenderer) strips domain
                                   prefixes from labels so 'causal::mechanism'
                                   renders as 'mechanism' in output text.

  3. Template diversity penalty  — C7 ResonanceMatcher applies a recency
                                   penalty so consecutive segments choose
                                   different sentence structures, producing
                                   varied, natural prose.

Before Phase 6 the rendered text looked like:
  'Unlike flow t0, the the underlying process is different in that
   the relevant factor. Unlike flow t1 — specifically regarding
   flow_t1 and flow_t2 — ...'

After Phase 6 it looks like:
  'Unlike mechanism, the mediation is different in that their
   interaction. While perturbation is true — specifically regarding
   perturbation and co occurrence —, co occurrence follows from
   different reasoning.'
"""
import numpy as np

from src.phase3.annealing_engine.experience import Experience
from src.phase5.pipeline.pipeline import GEOPipeline
from src.phase5.evaluation.evaluator import PipelineEvaluator

print("=== FLOW — Phase 6 Demo: Coherent Language Output ===\n")

# ── Build the full pipeline ────────────────────────────────────────────────────
pipeline = GEOPipeline(T0=0.8, lambda_=0.02, T_floor=0.02, flow_seed=42)
print(pipeline.summary())

# ── Seed the manifold with domain-labelled experiences ─────────────────────────
print("\n--- Seeding M(t) with domain-labelled experiences ---")
rng = np.random.default_rng(42)
M = pipeline.manifold

# Anchor positions from meaningful seed concepts
base_causal = M.position("causal::perturbation").copy()
base_mech   = M.position("causal::mechanism").copy()
base_logic  = M.position("domain::logical_entities").copy()
base_prob   = M.position("prob::maximal_uncertainty").copy()

domain_experiences = (
    [Experience(base_causal + rng.standard_normal(104) * 0.04,
                label=f"causal_study::event_{i}") for i in range(5)]
    + [Experience(base_mech + rng.standard_normal(104) * 0.04,
                  label=f"causal_study::mechanism_{i}") for i in range(5)]
    + [Experience(base_logic + rng.standard_normal(104) * 0.04,
                  label=f"logic_study::premise_{i}") for i in range(4)]
    + [Experience(base_prob + rng.standard_normal(104) * 0.04,
                  label=f"prob_study::estimate_{i}") for i in range(4)]
)
learn_results = pipeline.learn_batch(domain_experiences)
print(f"  Experiences processed : {len(learn_results)}")
print(f"  Mean novelty          : {sum(r.novelty for r in learn_results) / len(learn_results):.4f}")
print(f"  Temperature T(t)      : {pipeline.temperature:.4f}")
print(f"  Concepts on M(t)      : {pipeline.n_concepts}")

# ── Contrast judgments to shape the geometry ──────────────────────────────────
print("\n--- Contrast judgments (C4) ---")
cr1 = pipeline.contrast("causal_study::event_0", "causal_study::event_1", "same")
print(
    f"  SAME  event_0 / event_1       "
    f"Δdist={cr1.distance_change:+.4f}"
)
cr2 = pipeline.contrast("causal_study::event_0", "logic_study::premise_0", "different")
print(
    f"  DIFF  causal_event / premise   "
    f"Δdist={cr2.distance_change:+.4f}"
)

# ── Full pipeline queries — showing real concept labels in wave and text ───────
print("\n--- Full pipeline queries (C5 → C6 → C7) ---")
print("    The key improvement: WavePoint labels are now real manifold concepts,\n"
      "    not positional placeholders (flow_t0, flow_t1, ...).\n")

queries = [
    (base_causal, "what causes perturbation?"),
    (base_mech,   "describe the mechanism"),
    (base_logic,  "what is the logical structure?"),
    (base_prob,   "what is the uncertainty?"),
]

for vec, label in queries:
    result = pipeline.query(vec, label=label)

    # Show which real concepts appeared in the wave (not 'flow_t{i}')
    wave_labels = [
        p.label for p in result.wave.points
        if p.label and not p.label.startswith("flow_t")
    ]
    print(f"  query  : {label!r}")
    print(f"  steps  : {result.n_steps}  reason={result.termination_reason}")
    print(f"  wave Ψ : {len(result.wave.points)} points, "
          f"confidence={result.wave_confidence:.3f}")
    print(f"  concepts in wave : {wave_labels}")
    print(f"  render : confidence={result.confidence:.3f}  "
          f"flow_preserved={result.flow_preserved}")
    print(f"  text   :")
    for line in result.text.splitlines():
        print(f"    {line}")
    print()

print(f"  Total queries issued  : {pipeline.query_count}")

# ── Template diversity check ───────────────────────────────────────────────────
print("\n--- Template diversity (C7 ResonanceMatcher) ---")
print("    Verifying consecutive segments use different sentence structures.\n")
result_div = pipeline.query(base_causal, label="diversity_check")
matched_templates = [m.expression.text for m in result_div.output.matches]
print(f"  Segments matched          : {len(matched_templates)}")
print(f"  Unique templates used     : {len(set(matched_templates))}")
print(f"  Template diversity ratio  : "
      f"{len(set(matched_templates)) / max(len(matched_templates), 1):.2f}  "
      f"(1.0 = all different)")
print(f"  Templates selected:")
for i, t in enumerate(matched_templates):
    print(f"    [{i}] {t!r}")

# ── Label cleaning demonstration ──────────────────────────────────────────────
print("\n--- Label cleaning (C7 ExpressionRenderer._clean_label) ---")
print("    Domain prefixes are stripped before labels appear in output text.\n")
test_labels = [
    "causal::mechanism",
    "causal::co_occurrence",
    "domain::mathematical",
    "domain::logical_entities",
    "prob::maximal_uncertainty",
    "causal_study::mechanism_2",
]
from src.phase1.expression.renderer import ExpressionRenderer
renderer = ExpressionRenderer()
for lbl in test_labels:
    cleaned = renderer._clean_label(lbl)
    print(f"  {lbl!r:35s}  →  {cleaned!r}")

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\n--- Evaluation Framework ---")
evaluator = PipelineEvaluator(pipeline)

ev_q = evaluator.evaluate_query(base_causal, label="eval::causal")
print("  [Coherence — causal query]")
print(f"    overall score           : {ev_q.overall_score():.4f}")
print(f"    wave confidence         : {ev_q.coherence.wave_confidence:.4f}")
print(f"    render confidence       : {ev_q.coherence.render_confidence:.4f}")
print(f"    core fraction           : {ev_q.coherence.core_fraction:.4f}")
print(f"    trajectory steps        : {ev_q.coherence.trajectory_steps}")
print(f"    termination reason      : {ev_q.coherence.termination_reason}")

print("\n  [Causal direction]")
causal_m = evaluator.evaluate_causal_direction(
    cause_vec=base_causal, effect_vec=base_mech,
    cause_label="eval::cause", effect_label="eval::effect",
)
print(f"    causal_score            : {causal_m.causal_score:.4f}")
print(f"    forward steps           : {causal_m.forward_steps}")
print(f"    backward steps          : {causal_m.backward_steps}")

print("\n  [Locality check]")
loc = evaluator.evaluate_locality(rng.standard_normal(104), label="eval::locality")
print(f"    locality satisfied      : {loc.locality_satisfied}")
print(f"    n_nearby_moved          : {loc.n_nearby_moved}")
print(f"    n_distant_moved         : {loc.n_distant_moved}")
print(f"    max_distant_shift       : {loc.max_distant_shift:.2e}")

print("\n  [Novelty decay]")
decay = evaluator.evaluate_novelty_decay(base_causal, label="eval::decay", n_reps=5)
print(f"    novelty scores          : {[round(v, 4) for v in decay]}")
print(f"    monotonically dec.      : "
      f"{all(decay[i] >= decay[i+1] for i in range(len(decay)-1))}")

print("\n  [Full evaluation suite]")
suite = evaluator.run_suite(
    vectors=[base_causal, base_mech, base_logic, base_prob],
    labels=["suite::causal", "suite::mech", "suite::logic", "suite::prob"],
    novelty_reps=3,
)
d = suite.as_dict()
print(f"    n_queries               : {d['n_queries']}")
print(f"    mean_coherence          : {d['mean_coherence']:.4f}")
print(f"    mean_render_conf        : {d['mean_render_confidence']:.4f}")
print(f"    mean_wave_conf          : {d['mean_wave_confidence']:.4f}")
print(f"    mean_steps              : {d['mean_steps']:.1f}")
print(f"    terminations            : {d['termination_distribution']}")
print(f"    causal_score            : {d['causal_score']:.4f}")
print(f"    locality_satisfied      : {d['locality_satisfied']}")
print(f"    novelty_is_decaying     : {d['novelty_is_decaying']}")
print(f"    novelty_decay           : {[round(v, 4) for v in suite.novelty_decay]}")

print("\n=== Phase 6 demo complete ===")
