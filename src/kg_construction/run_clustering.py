"""
ClusteringPipeline — agglomerative clustering of KG entity/relation embeddings.

Reduces the embedding space from ~27K–41K raw entities to ~3K–4K cluster
centroids (threshold τ = 0.15, cosine distance, average linkage).

This makes patient-graph construction tractable: instead of 40K nodes,
training graphs have ≤ 4K cluster nodes.

Output files per graph directory:
  clusters_th015.json         — cluster_id → {nodes, embedding}
  clusters_inv_th015.json     — entity_id  → cluster_id
  clusters_rel_th015.json     — cluster_id → {relations, embedding}
  clusters_inv_rel_th015.json — relation_id → cluster_id
  ccscm_id2clus.json          — medical_code → cluster_id
  ccsproc_id2clus.json        — medical_code → cluster_id
  atc3_id2clus.json           — medical_code → cluster_id

Usage::

    from src.kg_construction import ClusteringPipeline

    pipe = ClusteringPipeline(threshold=0.15)
    pipe.run(
        graphs_dir="data/rocket_kg/graphs/merged/CCSCM_CCSPROC",
        triple_dirs={
            "ccscm":   "data/rocket_kg/graphs/condition/CCSCM",
            "ccsproc": "data/rocket_kg/graphs/procedure/CCSPROC",
        },
        out_dir="data/rocket_kg/clustering/ccscm_ccsproc",
    )
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """Agglomerative clustering of KG embeddings.

    Args:
        threshold : Cosine distance threshold for cluster formation.
        linkage   : Linkage criterion — "average" | "complete" | "ward".
    """

    def __init__(self, threshold: float = 0.15, linkage: str = "average"):
        self.threshold = threshold
        self.linkage = linkage

    # ------------------------------------------------------------------
    # Cluster embeddings
    # ------------------------------------------------------------------

    def cluster_embeddings(
        self, embeddings: np.ndarray
    ) -> tuple:
        """Run agglomerative clustering on an embedding matrix.

        Args:
            embeddings: [N, D] float32 embedding matrix.

        Returns:
            (labels, cluster_embeddings):
              labels [N,] — cluster assignment per entity
              cluster_embeddings [K, D] — mean embeddings per cluster
        """
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.threshold,
            linkage=self.linkage,
            metric="cosine",
            compute_distances=False,
        )
        labels = model.fit_predict(embeddings)
        n_clusters = labels.max() + 1

        cluster_emb = np.zeros((n_clusters, embeddings.shape[1]), dtype=np.float32)
        cluster_counts = np.zeros(n_clusters, dtype=int)

        for i, lbl in enumerate(labels):
            cluster_emb[lbl] += embeddings[i]
            cluster_counts[lbl] += 1

        for c in range(n_clusters):
            if cluster_counts[c] > 0:
                cluster_emb[c] /= cluster_counts[c]

        logger.info(f"Clustering: {len(embeddings)} → {n_clusters} clusters (τ={self.threshold})")
        return labels, cluster_emb

    # ------------------------------------------------------------------
    # Build cluster maps
    # ------------------------------------------------------------------

    def _build_cluster_map(
        self, labels: np.ndarray, id2name: Dict[int, str]
    ) -> tuple:
        """Build forward (cluster→members) and inverse (id→cluster) maps.

        Returns:
            (clusters, clusters_inv):
              clusters     : {cluster_id: {"nodes": [...], "embedding": [...]}}
              clusters_inv : {entity_id:  cluster_id}
        """
        clusters: Dict[str, dict] = {}
        clusters_inv: Dict[str, int] = {}

        for entity_id, cluster_id in enumerate(labels):
            c = int(cluster_id)
            e = str(entity_id)
            clusters_inv[e] = c
            if str(c) not in clusters:
                clusters[str(c)] = {"nodes": [], "embedding": None}
            clusters[str(c)]["nodes"].append(int(entity_id))

        return clusters, clusters_inv

    # ------------------------------------------------------------------
    # Code-to-cluster mapping
    # ------------------------------------------------------------------

    def _build_code2clus(
        self,
        triple_dir: str,
        ent2id: Dict[str, int],
        ent_labels: np.ndarray,
    ) -> Dict[str, int]:
        """Map each medical code file to its first entity's cluster.

        For each <code>.txt file in triple_dir, read the first entity string
        from the first valid triple, look up its embedding ID, and record the
        cluster it belongs to.

        Returns:
            code → cluster_id dict.
        """
        code2clus = {}
        for fpath in Path(triple_dir).glob("*.txt"):
            code = fpath.stem
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        entity = parts[0].strip().lower()
                        eid = ent2id.get(entity)
                        if eid is not None:
                            code2clus[code] = int(ent_labels[eid])
                            break
        return code2clus

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        graphs_dir: str,
        triple_dirs: Optional[Dict[str, str]] = None,
        out_dir: Optional[str] = None,
    ):
        """Run the full clustering pipeline.

        Args:
            graphs_dir  : Directory with ent2id.json + entity_embedding.pkl.
            triple_dirs : Dict mapping ontology_name → triple file directory,
                          used to build code→cluster maps.
                          e.g. {"ccscm": "graphs/condition/CCSCM"}.
            out_dir     : Where to save cluster files.  Defaults to graphs_dir.
        """
        graphs_path = Path(graphs_dir)
        out_path = Path(out_dir or graphs_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Load embeddings
        with open(graphs_path / "ent2id.json") as f:
            ent2id = json.load(f)
        with open(graphs_path / "id2ent.json") as f:
            id2ent = {int(k): v for k, v in json.load(f).items()}
        with open(graphs_path / "entity_embedding.pkl", "rb") as f:
            ent_emb = pickle.load(f)

        with open(graphs_path / "rel2id.json") as f:
            rel2id = json.load(f)
        with open(graphs_path / "id2rel.json") as f:
            id2rel = {int(k): v for k, v in json.load(f).items()}
        with open(graphs_path / "relation_embedding.pkl", "rb") as f:
            rel_emb = pickle.load(f)

        # --- Cluster entities ---
        ent_labels, ent_cluster_emb = self.cluster_embeddings(ent_emb)
        ent_clusters, ent_clusters_inv = self._build_cluster_map(ent_labels, id2ent)

        # Add embeddings to cluster map
        for cid, cdata in ent_clusters.items():
            cdata["embedding"] = ent_cluster_emb[int(cid)].tolist()

        # --- Cluster relations ---
        rel_labels, rel_cluster_emb = self.cluster_embeddings(rel_emb)
        rel_clusters, rel_clusters_inv = self._build_cluster_map(rel_labels, id2rel)
        for cid, cdata in rel_clusters.items():
            cdata["embedding"] = rel_cluster_emb[int(cid)].tolist()

        # --- Save cluster files ---
        suffix = f"th{str(self.threshold).replace('.', '')}"

        with open(out_path / f"clusters_{suffix}.json", "w") as f:
            json.dump(ent_clusters, f)
        with open(out_path / f"clusters_inv_{suffix}.json", "w") as f:
            json.dump(ent_clusters_inv, f)
        with open(out_path / f"clusters_rel_{suffix}.json", "w") as f:
            json.dump(rel_clusters, f)
        with open(out_path / f"clusters_inv_rel_{suffix}.json", "w") as f:
            json.dump(rel_clusters_inv, f)

        # Also save cluster embedding matrices for downstream use
        with open(out_path / f"map_cluster_{suffix}.pkl", "wb") as f:
            pickle.dump({"labels": ent_labels, "embeddings": ent_cluster_emb}, f)
        with open(out_path / f"map_cluster_rel_{suffix}.pkl", "wb") as f:
            pickle.dump({"labels": rel_labels, "embeddings": rel_cluster_emb}, f)

        # --- Code-to-cluster maps ---
        if triple_dirs:
            for ont_name, td in triple_dirs.items():
                code2clus = self._build_code2clus(td, ent2id, ent_labels)
                with open(out_path / f"{ont_name}_id2clus.json", "w") as f:
                    json.dump(code2clus, f)
                logger.info(f"  {ont_name}: {len(code2clus)} code→cluster mappings")

        logger.info(f"Clustering complete → {out_path}")
