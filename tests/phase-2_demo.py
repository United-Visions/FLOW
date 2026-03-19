"""Phase 2 end-to-end demo — Living Manifold + Contrast Engine."""
from src.phase1.seed_geometry import SeedGeometryEngine
from src.phase2.living_manifold.manifold import LivingManifold
from src.phase2.contrast_engine.engine import ContrastEngine, ContrastPair, JudgmentType

print("=== FLOW — Phase 2 Demo ===\n")

# ── Build M₀ (Phase 1) then wrap in Living Manifold ──────────────────────────
engine = SeedGeometryEngine()
M0 = engine.build()
M = LivingManifold(M0)
print(M.summary())

# ── READ operations ───────────────────────────────────────────────────────────
print("\n--- Living Manifold READ API ---")
p = M.position("causal::perturbation")
q = M.position("causal::direct_effect")
print(f"  distance(perturbation → direct_effect) : {M.distance(p, q):.4f}")
print(f"  causal_ancestry(p → q)                 : {M.causal_ancestry(p, q)}")
print(f"  region_type(p)                         : {M.region_type(p).value}")
print(f"  locality_radius(p)                     : {M.locality_radius(p):.4f}")
print(f"  confidence(p)                          : {M.confidence(p):.3f}")
print(f"  domain_of(p)                           : {M.domain_of(p)}")
nn = M.nearest(p, k=3)
print(f"  nearest(p, k=3)                        : {[l for l, _ in nn]}")
path = M.geodesic("causal::perturbation", "causal::direct_effect")
print(f"  geodesic path length                   : {len(path)} waypoints")

# ── WRITE: place a new concept ────────────────────────────────────────────────
print("\n--- WRITE: place new concept ---")
import numpy as np
vec = M.position("domain::causal_mechanisms").copy()
vec[0] += 0.3          # slight offset in base manifold
mp = M.place("concept::my_new_idea", vec)
print(f"  placed            : {mp.label}")
print(f"  n_points after    : {M.n_points}")
p_new = M.position("concept::my_new_idea")
print(f"  dist to origin    : {M.distance(p_new, vec):.6f}")

# ── WRITE: local deformation ──────────────────────────────────────────────────
print("\n--- WRITE: deform_local ---")
before = M.position("causal::perturbation").copy()
delta = np.zeros(104)
delta[0] = 0.05
n_affected = M.deform_local("causal::perturbation", delta)
after = M.position("causal::perturbation")
shift = float(np.linalg.norm(after - before))
print(f"  points affected   : {n_affected}")
print(f"  centre shifted by : {shift:.6f}")
print(f"  write ops so far  : {M.n_writes}")

# ── Contrast Engine ───────────────────────────────────────────────────────────
print("\n--- Contrast Engine ---")
ce = ContrastEngine(M, alpha=0.15, beta=0.15)

pairs = [
    ContrastPair("causal::perturbation", "causal::direct_effect",   JudgmentType.SAME),
    ContrastPair("causal::perturbation", "domain::epistemic",        JudgmentType.DIFFERENT),
    ContrastPair("causal::direct_effect","causal::downstream_effect", JudgmentType.SAME),
    ContrastPair("domain::mathematical", "domain::epistemic",        JudgmentType.DIFFERENT),
    ContrastPair("causal::perturbation", "causal::propagation",      JudgmentType.SAME),
]

for pair in pairs:
    result = ce.judge(pair.label_a, pair.label_b, pair.judgment)
    arrow = "↔ closer" if pair.judgment == JudgmentType.SAME else "↔ farther"
    print(f"  [{pair.judgment.value:9s}] {pair.label_a.split('::')[1]:20s} ↔ "
          f"{pair.label_b.split('::')[1]:20s} → Δdist "
          f"{result.distance_after - result.distance_before:+.4f}  {arrow}")

print(f"\n  total judgments   : {ce.n_judgments}")
print(f"  correct direction : {ce.correct_direction_rate():.1%}")
print()
print(ce.summary())

# ── Persistence diagram state ─────────────────────────────────────────────────
print("\n--- Persistence Diagram ---")
diag = ce._diagram
features = diag.get_persistent_features(min_lifetime=1)
print(f"  tracked pairs              : {len(diag._history)}")
print(f"  persistent features (>=1)  : {len(features)}")
corrections = diag.cluster_corrections()
print(f"  cluster corrections        : {len(corrections)}")
