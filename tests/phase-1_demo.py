"""Phase 1 end-to-end demo."""
from src.phase1.seed_geometry import SeedGeometryEngine
from src.phase1.expression import ExpressionRenderer, create_mock_wave

print("=== FLOW — Phase 1 Demo ===\n")

# Phase 1a: Build M0
engine = SeedGeometryEngine()
M0 = engine.build()

print()
print("--- M0 Query API ---")
p = M0.position("causal::perturbation")
q = M0.position("causal::direct_effect")
print(f"  distance(perturbation -> direct_effect) : {M0.distance(p, q):.4f}")
print(f"  causal ancestry (p -> q)               : {M0.causal_ancestry(p, q)}")
print(f"  confidence at perturbation             : {M0.confidence(p):.3f}")
print(f"  domain of perturbation                 : {M0.domain_of(p)}")
print(f"  curvature at perturbation              : {M0.curvature(p):.3f}")
print(f"  total seed points                      : {len(M0.seed_points)}")

print()
print("--- Expression Renderer (Phase 1b) ---")
renderer = ExpressionRenderer()
for theme in ["causation", "uncertainty", "contrast"]:
    wave = create_mock_wave(theme, seed=42)
    output = renderer.render(wave)
    print(f"\n  [{theme.upper()}] confidence={output.confidence:.2f}, segments={len(output.segments)}")
    print(f"  {output.text[:300]}")
