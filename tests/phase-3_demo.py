"""Phase 3 end-to-end demo — Annealing Engine (C3)."""
import numpy as np

from src.phase1.seed_geometry import SeedGeometryEngine
from src.phase2.living_manifold.manifold import LivingManifold
from src.phase3.annealing_engine import (
    AnnealingEngine,
    Experience,
    TemperatureSchedule,
    NoveltyEstimator,
)

print("=== FLOW — Phase 3 Demo ===\n")

# ── Build M₀ (Phase 1) then wrap in Living Manifold (Phase 2) ────────────────
engine_seed = SeedGeometryEngine()
M0 = engine_seed.build()
M = LivingManifold(M0)
print(M.summary())

# ── Temperature Schedule ──────────────────────────────────────────────────────
print("\n--- Temperature Schedule ---")
sched = TemperatureSchedule(T0=1.0, lambda_=0.05, T_floor=0.05)
print(f"  T(t=  0)  : {sched.temperature(  0.):.4f}")
print(f"  T(t= 20)  : {sched.temperature( 20.):.4f}")
print(f"  T(t= 50)  : {sched.temperature( 50.):.4f}")
print(f"  T(t=100)  : {sched.temperature(100.):.4f}")
print(f"  T(t=200)  : {sched.temperature(200.):.4f}")
print(f"  T_floor   : {sched.T_floor:.4f}  (never goes below)")
print(f"  radius@t=0  : {sched.locality_radius(base_radius=5.0, t=0.0):.4f}")
print(f"  radius@t=100: {sched.locality_radius(base_radius=5.0, t=100.0):.4f}")

# ── Novelty Estimator ─────────────────────────────────────────────────────────
print("\n--- Novelty Estimator ---")
est = NoveltyEstimator(sigma_scale=1.0)
pos = np.zeros(104)
no_neighbors     = est.estimate(pos, [],                             local_density=0.0)
known_neighbor   = est.estimate(pos, [pos.copy()],                   local_density=0.9)
distant_neighbor = est.estimate(pos, [np.ones(104) * 5.0],          local_density=0.0)
print(f"  no neighbours      → novelty {no_neighbors.score:.3f}   (maximum, unexplored)")
print(f"  identical + dense  → novelty {known_neighbor.score:.3f}   (minimum, well-known)")
print(f"  distant + sparse   → novelty {distant_neighbor.score:.3f}   (genuinely new)")

# ── Annealing Engine ──────────────────────────────────────────────────────────
print("\n--- AnnealingEngine setup ---")
anneal = AnnealingEngine(
    M,
    T0=1.0,
    lambda_=0.05,
    T_floor=0.05,
    k_neighbors=5,
    place_labeled=True,
)
print(f"  initial temperature : {anneal.temperature:.4f}")
print(f"  manifold points     : {M.n_points}")

# ── Single experience ─────────────────────────────────────────────────────────
print("\n--- Single experience ---")
vec_causal = M.position("causal::perturbation").copy()
vec_causal[0] += 0.2   # slight offset — nearby but distinct
r = anneal.process(Experience(vector=vec_causal, label="anneal::causal_variant_1"))
print(f"  label placed        : {r.placed_label}")
print(f"  resonance anchor    : {r.located_label}")
print(f"  novelty score       : {r.novelty:.3f}")
print(f"  temperature         : {r.temperature:.4f}")
print(f"  |δ| applied         : {r.delta_magnitude:.6f}")
print(f"  points affected     : {r.n_affected}")
print(f"  was_novel           : {r.was_novel}")

# ── Batch of experiences — self-organising stream ─────────────────────────────
print("\n--- Batch: 30 experiences from two conceptual clusters ---")
rng = np.random.default_rng(42)

# Cluster A — variations around causal::perturbation
base_a = M.position("causal::perturbation").copy()
cluster_a = [
    Experience(
        vector=base_a + rng.standard_normal(104) * 0.1,
        label=f"anneal::cluster_a_{i}",
        source="synthetic",
    )
    for i in range(15)
]

# Cluster B — variations around domain::mathematical
base_b = M.position("domain::mathematical").copy()
cluster_b = [
    Experience(
        vector=base_b + rng.standard_normal(104) * 0.1,
        label=f"anneal::cluster_b_{i}",
        source="synthetic",
    )
    for i in range(15)
]

T_before_batch = anneal.temperature
results = anneal.process_batch(cluster_a + cluster_b)
T_after_batch  = anneal.temperature

print(f"  processed           : {len(results)} experiences")
print(f"  T before batch      : {T_before_batch:.4f}")
print(f"  T after  batch      : {T_after_batch:.4f}  (cooled as expected)")
novel_count = sum(1 for r in results if r.was_novel)
print(f"  novel  (>0.5)       : {novel_count}/{len(results)}")
mean_nov = float(np.mean([r.novelty for r in results]))
print(f"  mean novelty        : {mean_nov:.3f}")
total_def = float(sum(r.delta_magnitude for r in results))
print(f"  total |δ| applied   : {total_def:.4f}")

# ── Manifold after annealing ──────────────────────────────────────────────────
print("\n--- Manifold state after annealing ---")
print(f"  points now          : {M.n_points}  (was 81 seed + placed concepts)")
print(f"  write ops           : {M.n_writes}")

# Check that placed concepts are retrievable
a_pos = M.position("anneal::cluster_a_0")
b_pos = M.position("anneal::cluster_b_0")
a_to_base_a = M.distance(a_pos, base_a)
b_to_base_b = M.distance(b_pos, base_b)
cross_dist   = M.distance(a_pos, b_pos)
print(f"  dist(cluster_a_0 → base_a) : {a_to_base_a:.4f}")
print(f"  dist(cluster_b_0 → base_b) : {b_to_base_b:.4f}")
print(f"  dist(cluster_a_0 → cluster_b_0) : {cross_dist:.4f}  (should be larger)")

# ── Reset temperature ─────────────────────────────────────────────────────────
print("\n--- Reset temperature ---")
T_before_reset = anneal.temperature
anneal.reset_temperature()
print(f"  T before reset      : {T_before_reset:.4f}")
print(f"  T after  reset      : {anneal.temperature:.4f}  (back to T₀+T_floor)")
print(f"  schedule time       : {anneal.t:.1f}  (reset to 0)")
print(f"  manifold points     : {M.n_points}  (unchanged by reset)")

# ── Full engine summary ───────────────────────────────────────────────────────
print("\n--- Engine summary ---")
print(anneal.summary())
