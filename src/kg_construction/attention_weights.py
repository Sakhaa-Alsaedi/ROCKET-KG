"""
AttentionWeightGenerator — task-specific attention weight initialisation.

Computes initial attention weights for each cluster node by measuring
semantic similarity between cluster embeddings and task-specific query terms.

The resulting weight vector is used to pre-initialise the α attention matrix
in the BAT / CADI / CAT models.

Normalisation strategy: cos_sim → sum-normalise → cube (sharpens distribution).

Usage::

    from src.kg_construction import AttentionWeightGenerator

    gen = AttentionWeightGenerator(openai_api_key="<ADD_YOUR_OPENAI_KEY_HERE>")  # ← Add your OpenAI API key here
    gen.generate_all(
        cluster_emb_file="data/rocket_kg/graphs/merged/CCSCM_CCSPROC/entity_embedding.pkl",
        out_dir="data/rocket_kg/attention_weights",
        tasks=["mortality", "readmission", "drugrec", "lenofstay"],
    )
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Per-task semantic query terms
TASK_TERMS: Dict[str, List[str]] = {
    "mortality": [
        "death", "mortality", "cause death", "lead to death",
        "high risk", "deadly", "fatal", "lethal",
    ],
    "readmission": [
        "rehospitalization", "readmission", "hospital return",
        "re-admission", "discharge complication",
    ],
    "drugrec": [
        "drug recommendation", "prescription", "drug", "medication",
        "treatment", "pharmacotherapy", "therapeutic agent",
    ],
    "lenofstay": [
        "length of stay", "bed days", "time in hospital",
        "hospital duration", "inpatient days",
    ],
}


class AttentionWeightGenerator:
    """Generate task-specific initial attention weights.

    Args:
        openai_api_key: OpenAI API key.
        model         : Embedding model name.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
    ):
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of text strings."""
        try:
            import openai
            openai.api_key = self.api_key
            resp = openai.Embedding.create(model=self.model, input=texts)
            return np.array([item["embedding"] for item in resp["data"]], dtype=np.float32)
        except ImportError:
            raise ImportError("openai required. pip install openai==0.27.4")

    def generate(
        self,
        cluster_embeddings: np.ndarray,
        task: str,
        task_terms: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute attention weights for a single task.

        Args:
            cluster_embeddings: [K, D] cluster centroid embeddings.
            task              : Task name (used to look up default terms).
            task_terms        : Override the default task terms.

        Returns:
            [K, 1] float32 weight vector, normalised to sum to 1 then cubed.
        """
        terms = task_terms or TASK_TERMS.get(task)
        if not terms:
            raise ValueError(f"Unknown task {task!r} and no task_terms provided.")

        term_emb = self._embed(terms)                      # [Q, D]

        # Normalise both sets of embeddings
        def _norm(m):
            norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-10
            return m / norms

        c_n = _norm(cluster_embeddings.astype(np.float32))
        t_n = _norm(term_emb)

        sim = c_n @ t_n.T                  # [K, Q]
        scores = sim.sum(axis=1)           # [K,] — sum over task terms

        # Normalise to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # Cube to sharpen: high-relevance clusters get much higher weight
        scores = scores ** 3
        scores = scores / (scores.sum() + 1e-10)

        return scores.reshape(-1, 1).astype(np.float32)

    def generate_all(
        self,
        cluster_emb_file: str,
        out_dir: str,
        tasks: Optional[List[str]] = None,
    ):
        """Generate and save weights for all tasks.

        Args:
            cluster_emb_file: Path to entity_embedding.pkl (cluster centroids).
            out_dir          : Output directory.
            tasks            : List of task names.  Default: all four.
        """
        tasks = tasks or list(TASK_TERMS.keys())
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(cluster_emb_file, "rb") as f:
            cluster_embeddings = pickle.load(f)

        logger.info(f"Loaded {len(cluster_embeddings)} cluster embeddings from {cluster_emb_file}")

        for task in tasks:
            weights = self.generate(cluster_embeddings, task)
            out_file = out_path / f"attention_weights_{task}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(weights, f)
            logger.info(f"  {task}: saved {weights.shape} weights → {out_file}")
