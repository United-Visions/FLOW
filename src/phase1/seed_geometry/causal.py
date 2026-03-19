"""
Causal Geometry
===============
Derives causal structure from first principles.

Source:   The definition of causation (Pearl's do-calculus)
Geometry: Directed acyclic graph embedded in continuous space.

Properties encoded
------------------
- Directionality  : causes precede effects as monotone τ-dimension curvature
- Asymmetry       : causal edges are geometrically irreversible
- Transitivity    : causal chains compose via path-distance on the embedded DAG
- Interventional  : do(X) vs observe(X) occupy spatially distinct sub-regions

Output: a directed Riemannian sub-manifold of dimension DIM_CAUSAL.
The first coordinate is always the "time-like" axis τ.

Design decision
---------------
The seed causal geometry is NOT derived from any data.  It derives from the
logical structure of causation itself: the archetypal causal relations that
must hold for any coherent concept of cause and effect.  Three layers of
Pearl's causal hierarchy are encoded as three dedicated sub-dimensions.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Dimensionality of the causal geometric fiber
DIM_CAUSAL = 16

# Pearl hierarchy layers — each gets a dedicated axis
LAYER_OBSERVATIONAL   = 0   # Layer 1: P(Y | X) — association
LAYER_INTERVENTIONAL  = 1   # Layer 2: P(Y | do(X)) — intervention
LAYER_COUNTERFACTUAL  = 2   # Layer 3: P(Y_x | X') — counterfactual

# τ-axis index (monotone causal time)
TAU_AXIS = 3


@dataclass
class CausalNode:
    """A node in the archetypal causal DAG."""
    name: str
    description: str
    tau: float          # position on the time-like axis (0 = earliest)
    layer: int          # Pearl hierarchy layer (0/1/2)
    concept_type: str   # 'force', 'event', 'state', 'mechanism', 'outcome'


@dataclass
class CausalGeometry:
    """
    The causal geometric fiber derived from Pearl's do-calculus.

    Attributes
    ----------
    dim : int
        Dimensionality of this fiber (DIM_CAUSAL).
    dag : nx.DiGraph
        The archetypal causal DAG — nodes are causal archetypes,
        edges are directed causal relationships.
    embeddings : dict[str, np.ndarray]
        Maps node name → DIM_CAUSAL-dimensional embedding vector.
    causal_metric : np.ndarray
        The asymmetric metric tensor for causal distance.
        Shape: (DIM_CAUSAL, DIM_CAUSAL).
    """

    dim: int = DIM_CAUSAL
    dag: nx.DiGraph = field(default_factory=nx.DiGraph)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    causal_metric: np.ndarray = field(default_factory=lambda: np.eye(DIM_CAUSAL))

    # ------------------------------------------------------------------ #
    # Construction                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls) -> "CausalGeometry":
        """
        Derive the causal geometry from first principles.

        Steps
        -----
        1. Define the archetypal causal nodes (universal causal concepts).
        2. Connect them into a DAG encoding necessary causal precedences.
        3. Embed the DAG geometrically using topological ordering.
        4. Compute the asymmetric causal metric tensor.
        """
        geo = cls()
        geo._define_archetypal_nodes()
        geo._define_archetypal_edges()
        geo._embed_dag()
        geo._compute_causal_metric()
        return geo

    def _define_archetypal_nodes(self) -> None:
        """
        Define the universal causal archetypes.

        These are the irreducible primitives of any causal reasoning:
        perturbation, propagation, mechanism, outcome, and intervention.
        """
        nodes: List[CausalNode] = [
            # ── Layer 0: Observational (what we see) ───────────────────
            CausalNode("initial_state",     "A system in steady state",         0.0, 0, "state"),
            CausalNode("perturbation",      "An unexpected change in a state",  0.1, 0, "event"),
            CausalNode("co_occurrence",     "Two events happening together",    0.2, 0, "event"),
            CausalNode("correlation",       "Statistical pattern detected",     0.3, 0, "state"),

            # ── Layer 1: Interventional (what we do) ───────────────────
            CausalNode("intervention",      "do(X) — force a variable to value",0.4, 1, "force"),
            CausalNode("mechanism",         "The physical process linking C→E", 0.5, 1, "mechanism"),
            CausalNode("propagation",       "Effect spreading through medium",  0.6, 1, "mechanism"),
            CausalNode("direct_effect",     "Immediate consequence of cause",   0.7, 1, "outcome"),
            CausalNode("downstream_effect", "Consequence of the direct effect", 0.8, 1, "outcome"),

            # ── Layer 2: Counterfactual (what would have been) ─────────
            CausalNode("counterfactual",    "Y_x: Y if X had been different",   0.5, 2, "state"),
            CausalNode("necessary_cause",   "C is necessary for E",             0.6, 2, "mechanism"),
            CausalNode("sufficient_cause",  "C alone produces E",               0.6, 2, "mechanism"),
            CausalNode("absent_cause",      "E occurs without C — reveals C",   0.7, 2, "event"),
            CausalNode("prevention",        "C prevents E",                      0.7, 2, "outcome"),

            # ── Meta-causal ─────────────────────────────────────────────
            CausalNode("confound",          "Common cause of C and E",          0.2, 0, "state"),
            CausalNode("mediation",         "C → mediator → E chain",           0.6, 1, "mechanism"),
        ]
        for n in nodes:
            self.dag.add_node(
                n.name,
                tau=n.tau,
                layer=n.layer,
                concept_type=n.concept_type,
                description=n.description,
            )

    def _define_archetypal_edges(self) -> None:
        """
        Add directed edges encoding mathematically necessary causal precedences.

        Each edge (A, B) means: A causally precedes B.
        These edges are derived from the *definition* of causation, not from data.
        """
        edges = [
            # Observational chain
            ("initial_state",     "perturbation"),
            ("perturbation",      "co_occurrence"),
            ("co_occurrence",     "correlation"),
            ("confound",          "co_occurrence"),

            # Interventional chain
            ("intervention",      "mechanism"),
            ("mechanism",         "propagation"),
            ("propagation",       "direct_effect"),
            ("direct_effect",     "downstream_effect"),
            ("perturbation",      "mechanism"),

            # Counterfactual branches
            ("intervention",      "counterfactual"),
            ("mechanism",         "necessary_cause"),
            ("mechanism",         "sufficient_cause"),
            ("direct_effect",     "necessary_cause"),
            ("sufficient_cause",  "direct_effect"),
            ("absent_cause",      "counterfactual"),
            ("mechanism",         "prevention"),
            ("mechanism",         "mediation"),
            ("mediation",         "direct_effect"),
        ]
        for src, dst in edges:
            if src in self.dag and dst in self.dag:
                self.dag.add_edge(src, dst)

        # Validate: must be a DAG
        assert nx.is_directed_acyclic_graph(self.dag), (
            "Causal archetype graph contains a cycle — violates causality by definition."
        )

    def _embed_dag(self) -> None:
        """
        Embed the causal DAG into DIM_CAUSAL-dimensional continuous space.

        Embedding strategy
        ------------------
        - Axis 0 (LAYER_OBSERVATIONAL)  : activation in observational layer
        - Axis 1 (LAYER_INTERVENTIONAL) : activation in interventional layer
        - Axis 2 (LAYER_COUNTERFACTUAL) : activation in counterfactual layer
        - Axis 3 (TAU_AXIS)             : topological sort depth (monotone)
        - Axes 4-7                      : outgoing causal degree, in-degree,
                                         path length to leaves, path length from roots
        - Axes 8-15                     : graph Laplacian eigenvector components
                                         (encodes relational topology of the DAG)
        """
        nodes = list(self.dag.nodes())
        n = len(nodes)

        # Topological sort depth as τ
        topo_order = {node: 0 for node in nodes}
        for node in nx.topological_sort(self.dag):
            preds = list(self.dag.predecessors(node))
            if preds:
                topo_order[node] = max(topo_order[p] for p in preds) + 1
        max_depth = max(topo_order.values()) or 1

        # Graph Laplacian eigenvectors (spectral embedding)
        A = nx.to_numpy_array(self.dag, nodelist=nodes, weight=None)
        A_sym = (A + A.T) / 2.0  # symmetrise for Laplacian
        D = np.diag(A_sym.sum(axis=1))
        L = D - A_sym
        eigvals, eigvecs = np.linalg.eigh(L)
        # Take eigenvectors 1..8 (skip the constant vector)
        spectral = eigvecs[:, 1:9] if eigvecs.shape[1] > 8 else eigvecs[:, 1:]
        # Pad if needed
        if spectral.shape[1] < 8:
            spectral = np.pad(spectral, ((0, 0), (0, 8 - spectral.shape[1])))

        for i, node in enumerate(nodes):
            data = self.dag.nodes[node]
            vec = np.zeros(self.dim)

            # Pearl hierarchy membership
            vec[LAYER_OBSERVATIONAL]   = 1.0 if data["layer"] == 0 else 0.0
            vec[LAYER_INTERVENTIONAL]  = 1.0 if data["layer"] == 1 else 0.0
            vec[LAYER_COUNTERFACTUAL]  = 1.0 if data["layer"] == 2 else 0.0

            # Monotone τ-axis
            vec[TAU_AXIS] = topo_order[node] / max_depth

            # Degree features
            vec[4] = self.dag.out_degree(node) / max(n - 1, 1)
            vec[5] = self.dag.in_degree(node)  / max(n - 1, 1)

            # Path distances
            try:
                lengths_from = nx.single_source_shortest_path_length(self.dag, node)
                vec[6] = max(lengths_from.values()) / max_depth
            except Exception:
                vec[6] = 0.0

            # Spectral embedding (axes 8..15)
            vec[8:8 + spectral.shape[1]] = spectral[i]

            self.embeddings[node] = vec

    def _compute_causal_metric(self) -> None:
        """
        Compute the asymmetric causal metric tensor.

        The causal metric is Riemannian with a direction-dependent correction:
        travel *with* causal direction is shorter than travel *against* it.

        Metric construction
        -------------------
        g = I + γ · (τ ⊗ τ)

        Where τ is the τ-axis unit vector and γ > 0 stretches the metric
        along the causal time direction, making retro-causal paths longer.
        """
        tau_vec = np.zeros(self.dim)
        tau_vec[TAU_AXIS] = 1.0
        gamma = 2.0   # stretch factor: retro-causal travel costs 3× more
        self.causal_metric = np.eye(self.dim) + gamma * np.outer(tau_vec, tau_vec)

    # ------------------------------------------------------------------ #
    # Query API                                                             #
    # ------------------------------------------------------------------ #

    def embed(self, node_name: str) -> np.ndarray:
        """Return the embedding vector for a named archetypal node."""
        if node_name not in self.embeddings:
            raise KeyError(f"Unknown causal archetype: '{node_name}'")
        return self.embeddings[node_name].copy()

    def causal_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Asymmetric causal distance from v1 to v2.

        d(v1 → v2) ≠ d(v2 → v1) when the causal τ-axis is involved.
        A path moving forward in τ is shorter; retro-causal paths are penalised.
        """
        diff = v2 - v1
        # Standard Riemannian distance
        base_dist = float(np.sqrt(diff @ self.causal_metric @ diff))
        # Asymmetry correction: penalise negative τ-direction travel
        delta_tau = v2[TAU_AXIS] - v1[TAU_AXIS]
        asymmetry = 1.0 + max(0.0, -delta_tau)   # penalty if going backward in τ
        return base_dist * asymmetry

    def causal_direction(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Unit vector pointing from v1 toward v2 in causal metric space.

        This is the seed for Flow Engine Force 2 (Causal Curvature).
        """
        diff = v2 - v1
        # Project through metric (Riemannian gradient direction)
        grad = self.causal_metric @ diff
        norm = np.linalg.norm(grad)
        if norm < 1e-12:
            return np.zeros(self.dim)
        return grad / norm

    def is_causal_ancestor(self, v1: np.ndarray, v2: np.ndarray) -> bool:
        """
        Return True if v1 is a likely causal ancestor of v2.

        Based on τ-axis ordering: v1 causally precedes v2 iff τ(v1) < τ(v2).
        """
        return float(v1[TAU_AXIS]) < float(v2[TAU_AXIS])

    def intervention_distance(self, v: np.ndarray) -> float:
        """
        Distance between the observational and interventional versions of v.

        do(X) and observe(X) are spatially distinct by their interventional axis.
        This is the gap between those two representations.
        """
        obs   = v.copy(); obs[LAYER_INTERVENTIONAL] = 0.0
        inter = v.copy(); inter[LAYER_INTERVENTIONAL] = 1.0
        return self.causal_distance(obs, inter)

    @property
    def node_names(self) -> List[str]:
        return list(self.dag.nodes())

    def summary(self) -> str:
        return (
            f"CausalGeometry(dim={self.dim}, "
            f"nodes={self.dag.number_of_nodes()}, "
            f"edges={self.dag.number_of_edges()}, "
            f"is_dag={nx.is_directed_acyclic_graph(self.dag)})"
        )
