"""
ROCKET Score — multi-dimensional knowledge-graph quality metric.

Five sub-scores evaluate different aspects of a ROCKET-KG:

  S1 — Structural Quality    : graph topology (density, connectivity, clustering)
  S2 — Semantic Coherence    : intra-cluster tightness vs. inter-cluster separation
  S3 — Task Relevance        : cosine alignment with task-specific query terms
  S4 — Causal Confidence     : ensemble agreement from causal discovery
  S5 — Clinical Coverage     : fraction of medical codes represented in the KG

The composite ROCKET Score is a weighted sum of S1–S5, defaulting to
equal weights.  All sub-scores are normalised to [0, 1].

Usage::

    from src.rocket_score import RocketScore
    import networkx as nx, numpy as np

    G = nx.karate_club_graph()
    emb = np.random.randn(G.number_of_nodes(), 64)
    labels = ["mortality", "sepsis", "ICU"]
    task_emb = np.random.randn(len(labels), 64)

    rs = RocketScore()
    result = rs.compute_all(
        graph=G,
        cluster_embeddings=emb,
        task_query_embeddings=task_emb,
        causal_adj_matrices=[np.random.randint(0, 2, (10, 10)) for _ in range(4)],
        total_codes=285,
        covered_codes=240,
    )
    print(result)   # {"S1": 0.71, "S2": 0.63, ..., "ROCKET": 0.68}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data class for results
# ---------------------------------------------------------------------------

@dataclass
class RocketScoreResult:
    S1: float  # Structural Quality
    S2: float  # Semantic Coherence
    S3: float  # Task Relevance
    S4: float  # Causal Confidence
    S5: float  # Clinical Coverage
    ROCKET: float  # Weighted composite
    weights: Dict[str, float] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"RocketScore("
            f"S1={self.S1:.3f}, S2={self.S2:.3f}, S3={self.S3:.3f}, "
            f"S4={self.S4:.3f}, S5={self.S5:.3f}, ROCKET={self.ROCKET:.3f})"
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "S1_structural": self.S1,
            "S2_semantic": self.S2,
            "S3_task_relevance": self.S3,
            "S4_causal": self.S4,
            "S5_coverage": self.S5,
            "ROCKET": self.ROCKET,
        }


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class RocketScore:
    """Compute the ROCKET Score for a knowledge graph.

    Args:
        weights (dict, optional): Per-sub-score weights.
            Default: equal weights (0.2 each).
    """

    DEFAULT_WEIGHTS = {"S1": 0.2, "S2": 0.2, "S3": 0.2, "S4": 0.2, "S5": 0.2}

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, (
            "Weights must sum to 1.0"
        )

    # ------------------------------------------------------------------
    # S1 — Structural Quality
    # ------------------------------------------------------------------

    def s1_structural(self, graph) -> float:
        """Compute structural quality of a NetworkX graph.

        Combines:
          - Normalised density (actual / maximum possible edges)
          - Largest connected component fraction
          - Average clustering coefficient (undirected approximation)

        Returns a score in [0, 1].
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for S1. pip install networkx")

        n = graph.number_of_nodes()
        if n == 0:
            return 0.0

        # --- Density ---
        density = nx.density(graph)

        # --- LCC fraction ---
        if graph.is_directed():
            ug = graph.to_undirected()
        else:
            ug = graph
        comps = list(nx.connected_components(ug))
        lcc_frac = max(len(c) for c in comps) / n if comps else 0.0

        # --- Clustering coefficient (undirected) ---
        try:
            avg_clust = nx.average_clustering(ug)
        except Exception:
            avg_clust = 0.0

        # Combine — equal third each
        score = (density + lcc_frac + avg_clust) / 3.0
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # S2 — Semantic Coherence
    # ------------------------------------------------------------------

    def s2_semantic(
        self,
        cluster_embeddings: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
    ) -> float:
        """Measure intra-cluster tightness of entity embeddings.

        Args:
            cluster_embeddings : [N, D] — one embedding per cluster centroid.
            cluster_assignments: [M,] — optional raw entity→cluster IDs for
                                 computing intra-cluster tightness directly.

        When ``cluster_assignments`` is None we approximate coherence as the
        mean pairwise cosine *distance* between cluster centroids (i.e. how
        spread out the clusters are — higher is better up to a point).

        Returns a score in [0, 1].
        """
        if cluster_embeddings is None or len(cluster_embeddings) == 0:
            return 0.0

        emb = cluster_embeddings.astype(np.float64)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
        emb_n = emb / norms

        # Mean pairwise cosine similarity among cluster centroids
        if len(emb_n) == 1:
            return 1.0

        # Use a random sample for large graphs to stay tractable
        max_sample = 512
        if len(emb_n) > max_sample:
            idx = np.random.choice(len(emb_n), max_sample, replace=False)
            emb_n = emb_n[idx]

        cos_sim = emb_n @ emb_n.T   # [K, K]
        # Exclude self-similarity diagonal
        mask = np.ones_like(cos_sim, dtype=bool)
        np.fill_diagonal(mask, False)
        mean_sim = cos_sim[mask].mean()

        # Transform: low mean similarity → well-separated clusters → better
        # Map to [0, 1] where ~0 mean sim → 1.0 score
        score = 1.0 - (mean_sim + 1.0) / 2.0  # cos_sim ∈ [-1,1] → [0,1] → invert
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # S3 — Task Relevance
    # ------------------------------------------------------------------

    def s3_task_relevance(
        self,
        cluster_embeddings: np.ndarray,
        task_query_embeddings: np.ndarray,
    ) -> float:
        """Measure alignment between KG cluster embeddings and task terms.

        For each cluster centroid we compute the max cosine similarity over
        all task query embeddings.  S3 = mean of these per-cluster max scores.

        Args:
            cluster_embeddings    : [K, D] cluster centroid embeddings.
            task_query_embeddings : [Q, D] task-term embeddings.

        Returns a score in [0, 1].
        """
        if len(cluster_embeddings) == 0 or len(task_query_embeddings) == 0:
            return 0.0

        def normalise(m):
            norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-10
            return m / norms

        c = normalise(cluster_embeddings.astype(np.float64))
        q = normalise(task_query_embeddings.astype(np.float64))

        sim = c @ q.T          # [K, Q]
        max_per_cluster = sim.max(axis=1)   # [K]
        score = max_per_cluster.mean()

        # Shift from [-1,1] to [0,1]
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # S4 — Causal Confidence
    # ------------------------------------------------------------------

    def s4_causal(self, causal_adj_matrices: Sequence[np.ndarray]) -> float:
        """Ensemble agreement score across causal discovery methods.

        Given K adjacency matrices (one per method, same shape [V, V]):
          - For each edge (i, j), compute fraction of methods that agree it exists.
          - S4 = mean pairwise agreement (Jaccard-like) across all method pairs.

        Args:
            causal_adj_matrices: List of binary [V, V] adjacency arrays.

        Returns a score in [0, 1].  Returns 1.0 for a single matrix.
        """
        if not causal_adj_matrices:
            return 0.0
        if len(causal_adj_matrices) == 1:
            return 1.0

        mats = [m.astype(bool) for m in causal_adj_matrices]
        K = len(mats)
        agreements = []
        for i in range(K):
            for j in range(i + 1, K):
                intersection = (mats[i] & mats[j]).sum()
                union = (mats[i] | mats[j]).sum()
                agreements.append(intersection / (union + 1e-10))

        return float(np.clip(np.mean(agreements), 0.0, 1.0))

    # ------------------------------------------------------------------
    # S5 — Clinical Coverage
    # ------------------------------------------------------------------

    def s5_coverage(self, total_codes: int, covered_codes: int) -> float:
        """Fraction of medical codes represented in the KG.

        Args:
            total_codes  : Total number of medical codes (e.g. 285 CCSCM codes).
            covered_codes: Number of codes with at least one KG triple.

        Returns a score in [0, 1].
        """
        if total_codes <= 0:
            return 0.0
        return float(np.clip(covered_codes / total_codes, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def compute_all(
        self,
        graph=None,
        cluster_embeddings: Optional[np.ndarray] = None,
        task_query_embeddings: Optional[np.ndarray] = None,
        causal_adj_matrices: Optional[Sequence[np.ndarray]] = None,
        total_codes: int = 0,
        covered_codes: int = 0,
        cluster_assignments: Optional[np.ndarray] = None,
    ) -> RocketScoreResult:
        """Compute all five sub-scores and the composite ROCKET Score.

        Any sub-score whose inputs are not provided defaults to 0.0.

        Args:
            graph                 : NetworkX graph (for S1).
            cluster_embeddings    : [K, D] centroid embeddings (for S2, S3).
            task_query_embeddings : [Q, D] task term embeddings (for S3).
            causal_adj_matrices   : List of [V, V] adjacency arrays (for S4).
            total_codes           : Total medical codes (for S5).
            covered_codes         : KG-covered codes (for S5).
            cluster_assignments   : [M,] entity→cluster IDs (optional, for S2).

        Returns:
            :class:`RocketScoreResult`
        """
        S1 = self.s1_structural(graph) if graph is not None else 0.0
        S2 = (
            self.s2_semantic(cluster_embeddings, cluster_assignments)
            if cluster_embeddings is not None
            else 0.0
        )
        S3 = (
            self.s3_task_relevance(cluster_embeddings, task_query_embeddings)
            if (cluster_embeddings is not None and task_query_embeddings is not None)
            else 0.0
        )
        S4 = self.s4_causal(causal_adj_matrices) if causal_adj_matrices else 0.0
        S5 = self.s5_coverage(total_codes, covered_codes)

        scores = {"S1": S1, "S2": S2, "S3": S3, "S4": S4, "S5": S5}
        composite = sum(self.weights[k] * v for k, v in scores.items())

        return RocketScoreResult(
            S1=S1, S2=S2, S3=S3, S4=S4, S5=S5,
            ROCKET=float(composite),
            weights=dict(self.weights),
        )
