"""
EmbeddingBuilder — retrieve and persist entity/relation embeddings.

Reads KG triple files produced by KGBuilder, collects all unique entity
and relation strings, and retrieves text-embedding-ada-002 vectors from
the OpenAI Embeddings API using parallel workers.

Output per graph directory (e.g. graphs/condition/CCSCM/):
  ent2id.json, id2ent.json, entity_embedding.pkl
  rel2id.json, id2rel.json, relation_embedding.pkl

Usage::

    from src.kg_construction import EmbeddingBuilder

    eb = EmbeddingBuilder(openai_api_key="<ADD_YOUR_OPENAI_KEY_HERE>", workers=20)  # ← Add your OpenAI API key here
    eb.build(graphs_dir="data/rocket_kg/graphs/condition/CCSCM")
    eb.merge(
        dirs=["graphs/condition/CCSCM", "graphs/procedure/CCSPROC"],
        out_dir="graphs/merged/CCSCM_CCSPROC",
    )
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    """Build and persist entity/relation embeddings from KG triple files.

    Args:
        openai_api_key : OpenAI key.  Reads ``OPENAI_API_KEY`` env if None.
        model          : Embedding model name.
        workers        : Parallel workers for API calls.
        batch_size     : Strings per API request.
    """

    EMBED_DIM = 1536  # text-embedding-ada-002 output size

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        workers: int = 20,
        batch_size: int = 100,
    ):
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.workers = workers
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Parse triples
    # ------------------------------------------------------------------

    @staticmethod
    def parse_triples(graphs_dir: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Collect all unique entities and relations from triple files.

        Returns:
            (ent2id, rel2id) dicts.
        """
        ent2id: Dict[str, int] = {}
        rel2id: Dict[str, int] = {}

        for fpath in Path(graphs_dir).glob("*.txt"):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    h, r, t = (p.strip().lower() for p in parts)
                    for e in (h, t):
                        if e not in ent2id:
                            ent2id[e] = len(ent2id)
                    if r not in rel2id:
                        rel2id[r] = len(rel2id)

        return ent2id, rel2id

    # ------------------------------------------------------------------
    # Retrieve embeddings
    # ------------------------------------------------------------------

    def retrieve_embeddings(self, strings: List[str]) -> np.ndarray:
        """Retrieve embeddings for a list of strings in parallel.

        Args:
            strings: Unique strings to embed.

        Returns:
            [N, EMBED_DIM] float32 array (same order as input).
        """
        emb = np.zeros((len(strings), self.EMBED_DIM), dtype=np.float32)
        idx_map = {s: i for i, s in enumerate(strings)}

        batches = [
            strings[i: i + self.batch_size]
            for i in range(0, len(strings), self.batch_size)
        ]

        def _embed_batch(batch):
            return self._call_openai(batch)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_embed_batch, b): b for b in batches}
            for fut in as_completed(futures):
                batch = futures[fut]
                try:
                    vecs = fut.result()
                    for s, v in zip(batch, vecs):
                        emb[idx_map[s]] = v
                except Exception as e:
                    logger.warning(f"Embedding batch failed: {e}")

        return emb

    # ------------------------------------------------------------------
    # Build (single directory)
    # ------------------------------------------------------------------

    def build(self, graphs_dir: str, out_dir: Optional[str] = None):
        """Build embeddings for all triples in a directory.

        Saves ent2id.json, id2ent.json, entity_embedding.pkl,
        rel2id.json, id2rel.json, relation_embedding.pkl to out_dir
        (default: same as graphs_dir).

        Args:
            graphs_dir: Directory containing .txt triple files.
            out_dir   : Where to write output files.  Defaults to graphs_dir.
        """
        out_path = Path(out_dir or graphs_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ent2id, rel2id = self.parse_triples(graphs_dir)
        logger.info(f"{graphs_dir}: {len(ent2id)} entities, {len(rel2id)} relations")

        id2ent = {v: k for k, v in ent2id.items()}
        id2rel = {v: k for k, v in rel2id.items()}

        ent_strings = [id2ent[i] for i in range(len(ent2id))]
        rel_strings = [id2rel[i] for i in range(len(rel2id))]

        ent_emb = self.retrieve_embeddings(ent_strings)
        rel_emb = self.retrieve_embeddings(rel_strings)

        # Persist
        with open(out_path / "ent2id.json", "w") as f:
            json.dump(ent2id, f)
        with open(out_path / "id2ent.json", "w") as f:
            json.dump(id2ent, f)
        with open(out_path / "rel2id.json", "w") as f:
            json.dump(rel2id, f)
        with open(out_path / "id2rel.json", "w") as f:
            json.dump(id2rel, f)
        with open(out_path / "entity_embedding.pkl", "wb") as f:
            pickle.dump(ent_emb, f)
        with open(out_path / "relation_embedding.pkl", "wb") as f:
            pickle.dump(rel_emb, f)

        logger.info(f"Saved embeddings to {out_path}")

    # ------------------------------------------------------------------
    # Merge (combine multiple type embeddings)
    # ------------------------------------------------------------------

    def merge(self, dirs: List[str], out_dir: str):
        """Merge embeddings from multiple directories (types) with offset IDs.

        Useful for combining condition + procedure + drug embeddings into a
        single global namespace (CCSCM_CCSPROC_ATC3).

        Args:
            dirs   : List of directories, each with ent2id.json + embeddings.pkl.
            out_dir: Output directory for merged files.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        merged_ent2id: Dict[str, int] = {}
        merged_rel2id: Dict[str, int] = {}
        ent_matrices = []
        rel_matrices = []

        for d in dirs:
            d = Path(d)

            with open(d / "ent2id.json") as f:
                local_ent2id = json.load(f)
            with open(d / "rel2id.json") as f:
                local_rel2id = json.load(f)
            with open(d / "entity_embedding.pkl", "rb") as f:
                local_ent_emb = pickle.load(f)
            with open(d / "relation_embedding.pkl", "rb") as f:
                local_rel_emb = pickle.load(f)

            ent_offset = len(merged_ent2id)
            rel_offset = len(merged_rel2id)

            for name, idx in local_ent2id.items():
                if name not in merged_ent2id:
                    merged_ent2id[name] = idx + ent_offset

            for name, idx in local_rel2id.items():
                if name not in merged_rel2id:
                    merged_rel2id[name] = idx + rel_offset

            ent_matrices.append(local_ent_emb)
            rel_matrices.append(local_rel_emb)

        merged_ent_emb = np.concatenate(ent_matrices, axis=0)
        merged_rel_emb = np.concatenate(rel_matrices, axis=0)

        merged_id2ent = {v: k for k, v in merged_ent2id.items()}
        merged_id2rel = {v: k for k, v in merged_rel2id.items()}

        with open(out_path / "ent2id.json", "w") as f:
            json.dump(merged_ent2id, f)
        with open(out_path / "id2ent.json", "w") as f:
            json.dump(merged_id2ent, f)
        with open(out_path / "rel2id.json", "w") as f:
            json.dump(merged_rel2id, f)
        with open(out_path / "id2rel.json", "w") as f:
            json.dump(merged_id2rel, f)
        with open(out_path / "entity_embedding.pkl", "wb") as f:
            pickle.dump(merged_ent_emb, f)
        with open(out_path / "relation_embedding.pkl", "wb") as f:
            pickle.dump(merged_rel_emb, f)

        logger.info(
            f"Merged {len(dirs)} dirs → {len(merged_ent2id)} entities, "
            f"{len(merged_rel2id)} relations → {out_path}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_openai(self, strings: List[str]) -> List[np.ndarray]:
        """Call OpenAI Embeddings API for a batch of strings."""
        try:
            import openai
            openai.api_key = self.api_key
            resp = openai.Embedding.create(model=self.model, input=strings)
            return [np.array(item["embedding"], dtype=np.float32) for item in resp["data"]]
        except ImportError:
            raise ImportError("openai package required. pip install openai==0.27.4")
