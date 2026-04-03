"""
Evaluator — wraps model inference + metric computation into a single object.

Usage::

    from src.evaluation import Evaluator

    ev = Evaluator(task="mortality", device="cuda")
    metrics = ev.evaluate(model, dataloader, loss_fn)
    print(metrics)   # {"roc_auc": 0.71, "f1": 0.54, ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .metrics import compute_task_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs inference on a DataLoader and computes task-specific metrics.

    Args:
        task   : "mortality" | "readmission" | "drugrec" | "lenofstay".
        device : Torch device string.
        threshold: Decision threshold (binary / multi-label tasks).
    """

    def __init__(
        self,
        task: str,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self.task = task
        self.device = torch.device(device)
        self.threshold = threshold
        self._is_multilabel = task == "drugrec"
        self._is_multiclass = task == "lenofstay"

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Run evaluation on a DataLoader.

        The DataLoader must yield batches in the following format:
          batch = (node_ids, rel_ids, edge_index, batch_vec, visit_node,
                   ehr_nodes, labels)

        For non-graph models (RNN / Transformer / RETAIN / MLP), override
        :meth:`_predict_batch` in a subclass.

        Args:
            model      : Trained PyTorch model (set to eval mode internally).
            dataloader : Iterable yielding batches.
            loss_fn    : Optional — if provided, also returns average loss.

        Returns:
            Dict of metric names → float values.
        """
        model.eval()
        model.to(self.device)

        all_logits, all_labels = [], []
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            logits, labels = self._predict_batch(model, batch)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            if loss_fn is not None:
                loss = loss_fn(logits, labels.float().to(self.device))
                total_loss += loss.item()
            n_batches += 1

        if not all_logits:
            logger.warning("Evaluator received empty dataloader.")
            return {}

        logits = torch.cat(all_logits, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        # Convert logits → probabilities
        if self._is_multiclass:
            y_score = self._softmax(logits)
        elif self._is_multilabel:
            y_score = self._sigmoid(logits)
        else:
            y_score = self._sigmoid(logits).squeeze(-1)
            if y_score.ndim > 1 and y_score.shape[1] == 1:
                y_score = y_score[:, 0]

        metrics = compute_task_metrics(self.task, labels, y_score, self.threshold)

        if loss_fn is not None and n_batches > 0:
            metrics["loss"] = total_loss / n_batches

        return metrics

    def _predict_batch(self, model, batch):
        """Extract logits and labels from a batch.

        Override for custom batch formats.
        """
        (node_ids, rel_ids, edge_index, batch_vec,
         visit_node, ehr_nodes, labels) = batch

        node_ids = node_ids.to(self.device)
        rel_ids = rel_ids.to(self.device)
        edge_index = edge_index.to(self.device)
        batch_vec = batch_vec.to(self.device)
        visit_node = visit_node.to(self.device)
        ehr_nodes = [e.to(self.device) for e in ehr_nodes]

        logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)
        return logits, labels.to(self.device)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def save_results(self, metrics: Dict[str, float], path: str):
        """Save metric dict to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {path}")
