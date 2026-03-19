"""Phase 4 end-to-end demo — Flow Engine (C5) and Resonance Layer (C6)."""
import numpy as np

from src.phase1.seed_geometry import SeedGeometryEngine
from src.phase1.expression.renderer import ExpressionRenderer
from src.phase2.living_manifold.manifold import LivingManifold
from src.phase3.annealing_engine import AnnealingEngine, Experience
from src.phase4.flow_engine import FlowEngine, Query
from src.phase4.flow_engine.forces import ForceComputer
from src.phase4.flow_engine.sde import SDESolver
from src.phase4.resonance_layer import ResonanceLayer

print("=== FLOW — Phase 4 Demo ===\n")

# ── Build M₀ → Living Manifold → Annealing enrichment ────────────────────────
seed_engine = SeedGeometryEngine()
M0 = seed_engine.build()
M = LivingManifold(M0)

# Enrich the manifold with a small batch of experiences so the Flow Engine
# has richer geometry to navigate.
anneal = AnnealingEngine(M, T0=0.5, lambda_=0.02, T_floor=0.02)
rng = np.random.default_rng(42)
base_causal = M.position("causal::perturbation").copy()
base_math   = M.position("domain::mathematical").copy()
experiences = (
    [Experience(base_causal + rng.standard_normal(104) * 0.05,
                label=f"demo::causal_{i}") for i in range(8)]
    + [Experience(base_math + rng.standard_normal(104) * 0.05,
                  label=f"demo::math_{i}") for i in range(8)]
)
anneal.process_batch(experiences)

print(M.summary())

# ── Force Inspection ──────────────────────────────────────────────────────────
print("\n--- Force Computer ---")
fc = ForceComputer()
pos = M.position("causal::perturbation").copy()
vel = np.zeros(104)
f_grav = fc.semantic_gravity(pos, M)
f_cau  = fc.causal_curvature(pos, vel, M)
f_mom  = fc.contextual_momentum(vel * 0)
f_rep  = fc.contrast_repulsion(pos, M)
print(f"  position            : causal::perturbation")
print(f"  ‖F_gravity‖         : {float(np.linalg.norm(f_grav)):.4f}")
print(f"  ‖F_causal‖          : {float(np.linalg.norm(f_cau)):.4f}")
print(f"  ‖F_momentum‖        : {float(np.linalg.norm(f_mom)):.4f}  (zero init velocity)")
print(f"  ‖F_repulsion‖       : {float(np.linalg.norm(f_rep)):.4f}")

# ── SDE Solver ────────────────────────────────────────────────────────────────
print("\n--- SDE Solver ---")
sde = SDESolver(dt=0.05, diffusion_scale=0.05, rng=np.random.default_rng(0))
drift = fc.combined_drift(pos, vel, M)
new_pos, new_vel = sde.step(pos, drift, M)
sigma = sde.diffusion_at(pos, M)
print(f"  dt                  : {sde.dt}")
print(f"  diffusion_scale     : {sde.diffusion_scale}")
print(f"  σ(P) at seed point  : {sigma:.6f}  (low — dense, crystallised region)")
print(f"  step ‖Δposition‖    : {float(np.linalg.norm(new_pos - pos)):.6f}")
print(f"  step ‖velocity‖     : {float(np.linalg.norm(new_vel)):.6f}")

# ── Flow Engine — single query ────────────────────────────────────────────────
print("\n--- Flow Engine (single query) ---")
fe = FlowEngine(M, max_steps=150, dt=0.05, seed=7)
query = Query(
    vector=M.position("causal::perturbation").copy(),
    label="what causes perturbation?",
)
trajectory = fe.flow(query)

print(f"  query label         : {query.label}")
print(f"  n_steps             : {len(trajectory)}")
print(f"  total flow time     : {trajectory.total_time:.3f}")
print(f"  termination reason  : {trajectory.termination_reason}")
print(f"  mean speed          : {trajectory.mean_speed:.6f}")
print(f"  mean curvature      : {trajectory.mean_curvature:.4f}")

# Show first and last positions
p_first = trajectory.steps[0].position
p_last  = trajectory.steps[-1].position
dist_first_last = float(np.linalg.norm(p_last - p_first))
print(f"  ‖P_last − P_first‖  : {dist_first_last:.4f}  (traversal distance)")

# ── Three conceptual queries ──────────────────────────────────────────────────
print("\n--- Three conceptual queries ---")
queries_to_test = [
    ("causal::perturbation",   "causal query"),
    ("domain::mathematical",   "mathematical query"),
    ("causal::mechanism",      "mechanism query"),
]
for label, description in queries_to_test:
    q = Query(vector=M.position(label).copy(), label=description)
    fe_q = FlowEngine(M, max_steps=80, seed=42)
    traj = fe_q.flow(q)
    print(
        f"  {description:<24} → steps={len(traj):>3}  "
        f"reason={traj.termination_reason:<20}  "
        f"mean_speed={traj.mean_speed:.4f}"
    )

# ── Resonance Layer ───────────────────────────────────────────────────────────
print("\n--- Resonance Layer ---")
rl = ResonanceLayer(M, resonance_radius=0.5, harmonic_tolerance=0.15)
wave = rl.accumulate(trajectory)

print(f"  trajectory steps    : {len(trajectory)}")
print(f"  wave points (Ψ>0)  : {len(wave.points)}")
print(f"  total energy ∫Ψ    : {wave.total_energy:.4f}")
print(f"  wave confidence     : {wave.mean_confidence():.3f}")
print(f"  wave uncertainty    : {wave.mean_uncertainty():.3f}")

if wave.peak:
    peak = wave.peak
    print(f"  peak point label    : {peak.label}")
    print(f"  peak amplitude      : {peak.amplitude:.4f}")
    print(f"  peak τ (causal time): {peak.tau:.3f}")

core = wave.confident_core(threshold=0.4)
print(f"  confident core (≥0.4 norm amp) : {len(core)} points")

# Metadata from C6
meta = wave.metadata
print(f"  n_trajectory_steps  : {meta['n_trajectory_steps']}")
print(f"  termination_reason  : {meta['termination_reason']}")
print(f"  mean_speed (traj)   : {meta['trajectory_mean_speed']:.4f}")

# ── Query echo ────────────────────────────────────────────────────────────────
print("\n--- Query echo in wave ---")
echo = wave.query_echo
if echo:
    print(f"  label               : {echo.label}")
    print(f"  amplitude           : {echo.amplitude:.4f}  (weak — query echo only)")
    print(f"  τ                   : {echo.tau:.3f}  (origin of the flow)")

# ── Full pipeline: Flow → Resonance → Expression ──────────────────────────────
print("\n--- Full pipeline: C5 → C6 → C7 ---")
full_queries = [
    ("causal::mechanism", "mechanism"),
    ("causal::intervention", "intervention"),
]
renderer = ExpressionRenderer()

for label, description in full_queries:
    q = Query(vector=M.position(label).copy(), label=description)
    fe_full = FlowEngine(M, max_steps=60, seed=99)
    traj_full  = fe_full.flow(q)
    wave_full  = rl.accumulate(traj_full)
    output     = renderer.render(wave_full)
    print(f"  query     : {description}")
    print(f"  steps     : {len(traj_full)}, reason={traj_full.termination_reason}")
    print(f"  wave pts  : {len(wave_full.points)}, energy={wave_full.total_energy:.3f}")
    print(f"  rendered  : \"{output.text}\"")
    print(f"  confidence: {output.confidence:.3f}")
    print()
