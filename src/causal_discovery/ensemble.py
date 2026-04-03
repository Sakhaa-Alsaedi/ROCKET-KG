"""
Ensemble Causal Discovery

Runs multiple causal discovery algorithms on the same data and aggregates
their inferred causal graphs via a majority-vote ensemble:

  Method     | Package         | Type
  -----------|-----------------|---------------------------
  PC         | causal-learn    | Constraint-based (CI tests)
  LiNGAM     | lingam          | Linear non-Gaussian acyclic
  NOTEARS    | (implemented)   | Gradient-based DAG learning
  GOLEM      | (implemented)   | Score-based DAG learning

Optional methods (only run if packages are installed) can be toggled with
the ``methods`` constructor argument.

Usage::

    import numpy as np
    from src.causal_discovery import CausalEnsemble

    X = np.random.randn(500, 10)        # 500 samples, 10 variables
    ce = CausalEnsemble(methods=["notears", "golem"])
    result = ce.fit(X)

    print(result.adjacency)             # [10, 10] soft edge weights
    print(result.binary_adjacency)      # thresholded [10, 10] bool array
    print(result.agreement)             # fraction of methods that agreed on each edge
    print(result.s4_score)              # ROCKET S4 score
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CausalResult:
    """Aggregated output of the causal ensemble."""
    adjacency: np.ndarray         # [V, V] soft edge weights (mean over methods)
    binary_adjacency: np.ndarray  # [V, V] bool — majority vote
    agreement: np.ndarray         # [V, V] fraction of methods voting for each edge
    method_results: Dict[str, np.ndarray]  # raw binary adj per method
    s4_score: float               # ROCKET S4 causal confidence

    def __repr__(self):
        V = self.adjacency.shape[0]
        n_edges = int(self.binary_adjacency.sum())
        return (
            f"CausalResult(nodes={V}, edges={n_edges}, "
            f"methods={list(self.method_results)}, s4={self.s4_score:.3f})"
        )


# ---------------------------------------------------------------------------
# NOTEARS  (standalone, no external dependency)
# ---------------------------------------------------------------------------

def _notears(X: np.ndarray, lambda1: float = 0.1, max_iter: int = 100) -> np.ndarray:
    """Minimal NOTEARS implementation (Zheng et al., NeurIPS 2018).

    Solves::

        min  (1/2n) ||X - X W^T||²_F + λ||W||₁
        s.t. tr(e^{W⊙W}) − d = 0    (DAG constraint)

    Uses the augmented Lagrangian method with gradient descent.

    Args:
        X       : [N, d] data matrix (zero-mean recommended).
        lambda1 : L1 regularisation strength.
        max_iter: Maximum outer iterations.

    Returns:
        W [d, d] estimated weighted adjacency (W_ij ≠ 0 → i → j).
    """
    n, d = X.shape
    W = np.zeros((d, d))

    def _loss(W_flat):
        W_ = W_flat.reshape(d, d)
        M = X - X @ W_.T
        loss = 0.5 / n * (M ** 2).sum()
        grad = -1.0 / n * (M.T @ X)
        # L1 subgradient
        loss += lambda1 * np.abs(W_).sum()
        grad += lambda1 * np.sign(W_)
        return loss, grad.ravel()

    def _h(W_):
        """DAG penalty: tr(e^{W⊙W}) − d."""
        M = W_ * W_
        E = np.linalg.matrix_power(
            np.eye(d) + M / d, d
        )          # approximation: (I + M/d)^d ≈ e^M
        return np.trace(E) - d, E.T * 2 / d  # value, grad

    rho, alpha = 1.0, 0.0
    h_tol = 1e-8
    rho_max = 1e16
    lr = 1e-3

    for _ in range(max_iter):
        # Augmented Lagrangian gradient step
        for __ in range(300):
            h_val, h_grad = _h(W)
            loss_val, loss_grad = _loss(W.ravel())
            total_grad = (loss_grad.reshape(d, d)
                          + (alpha + rho * h_val) * h_grad)
            W = W - lr * total_grad
            W[np.abs(W) < 1e-6] = 0.0     # threshold small values

        h_val, _ = _h(W)
        if abs(h_val) <= h_tol:
            break
        alpha += rho * h_val
        rho = min(rho * 10, rho_max)

    W[np.abs(W) < 0.3] = 0.0   # final sparsification threshold
    return W


# ---------------------------------------------------------------------------
# GOLEM  (standalone, simplified score-based)
# ---------------------------------------------------------------------------

def _golem(X: np.ndarray, lambda1: float = 0.02, lambda2: float = 5.0,
           equal_variances: bool = True, max_iter: int = 100) -> np.ndarray:
    """Simplified GOLEM (Ng et al., NeurIPS 2020).

    Uses equal-variance log-likelihood + DAG penalty.

    Args:
        X              : [N, d] data matrix.
        lambda1        : L1 regularisation.
        lambda2        : DAG constraint weight.
        equal_variances: Use equal-variance likelihood.
        max_iter       : Gradient descent iterations.

    Returns:
        W [d, d] estimated weighted adjacency.
    """
    n, d = X.shape
    W = np.zeros((d, d))
    lr = 1e-3

    def _score(W_):
        """GOLEM score (EV likelihood + penalties)."""
        B = np.eye(d) - W_.T
        XB = X @ B
        if equal_variances:
            score = n * d / 2 * np.log((XB ** 2).mean())
        else:
            score = 0.5 * n * np.log(((XB ** 2).sum(axis=0))).sum()

        # Log|det(I - W)|  DAG term (negative log-likelihood contribution)
        try:
            score -= n * np.log(np.abs(np.linalg.det(B)) + 1e-10)
        except Exception:
            pass

        # L1 penalty
        score += lambda1 * np.abs(W_).sum()
        return score

    for _ in range(max_iter):
        eps = 1e-5
        grad = np.zeros_like(W)
        for i in range(d):
            for j in range(d):
                dW = np.zeros_like(W)
                dW[i, j] = eps
                grad[i, j] = (_score(W + dW) - _score(W - dW)) / (2 * eps)
        W = W - lr * grad
        W[np.abs(W) < 1e-6] = 0.0

    W[np.abs(W) < 0.3] = 0.0
    return W


# ---------------------------------------------------------------------------
# Wrappers for optional external packages
# ---------------------------------------------------------------------------

def _run_pc(X: np.ndarray) -> Optional[np.ndarray]:
    """Run PC algorithm via causal-learn."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
        cg = pc(X, alpha=0.05, indep_test=fisherz)
        adj = (cg.G.graph != 0).astype(int)
        return adj
    except ImportError:
        logger.warning("causal-learn not installed; skipping PC. pip install causal-learn")
        return None
    except Exception as e:
        logger.warning(f"PC algorithm failed: {e}")
        return None


def _run_lingam(X: np.ndarray) -> Optional[np.ndarray]:
    """Run DirectLiNGAM via lingam."""
    try:
        import lingam
        model = lingam.DirectLiNGAM()
        model.fit(X)
        adj = (np.abs(model.adjacency_matrix_) > 0.1).astype(int)
        return adj
    except ImportError:
        logger.warning("lingam not installed; skipping LiNGAM. pip install lingam")
        return None
    except Exception as e:
        logger.warning(f"LiNGAM algorithm failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CausalEnsemble
# ---------------------------------------------------------------------------

class CausalEnsemble:
    """Ensemble causal discovery across multiple methods.

    Args:
        methods (list): Subset of ["pc", "lingam", "notears", "golem"].
            Default: all four (optional external deps used when available).
        threshold (float): Majority-vote threshold. Edge is included if
            ``agreement >= threshold``.  Default: 0.5 (strict majority).
        notears_lambda (float): NOTEARS L1 regularisation.
        golem_lambda1 (float): GOLEM L1 regularisation.
    """

    ALL_METHODS = ["pc", "lingam", "notears", "golem"]

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        threshold: float = 0.5,
        notears_lambda: float = 0.1,
        golem_lambda1: float = 0.02,
    ):
        self.methods = self.ALL_METHODS if methods is None else methods
        self.threshold = threshold
        self.notears_lambda = notears_lambda
        self.golem_lambda1 = golem_lambda1

    def fit(self, X: np.ndarray) -> CausalResult:
        """Run causal discovery on data matrix X.

        Args:
            X: [N, d] data matrix. Standardise columns before passing.

        Returns:
            :class:`CausalResult`
        """
        n, d = X.shape
        raw: Dict[str, np.ndarray] = {}

        if "pc" in self.methods:
            result = _run_pc(X)
            if result is not None:
                raw["pc"] = (result != 0).astype(int)

        if "lingam" in self.methods:
            result = _run_lingam(X)
            if result is not None:
                raw["lingam"] = result

        if "notears" in self.methods:
            try:
                W = _notears(X, lambda1=self.notears_lambda)
                raw["notears"] = (np.abs(W) > 0.1).astype(int)
            except Exception as e:
                logger.warning(f"NOTEARS failed: {e}")

        if "golem" in self.methods:
            try:
                W = _golem(X, lambda1=self.golem_lambda1)
                raw["golem"] = (np.abs(W) > 0.1).astype(int)
            except Exception as e:
                logger.warning(f"GOLEM failed: {e}")

        if not raw:
            # Fallback: empty adjacency
            adj = np.zeros((d, d))
            return CausalResult(
                adjacency=adj,
                binary_adjacency=adj.astype(bool),
                agreement=adj,
                method_results=raw,
                s4_score=0.0,
            )

        # Stack and aggregate
        stack = np.stack(list(raw.values()), axis=0)   # [K, d, d]
        agreement = stack.mean(axis=0)                 # [d, d] ∈ [0,1]
        binary = (agreement >= self.threshold)

        # ROCKET S4 — pairwise Jaccard agreement
        mats = list(raw.values())
        K = len(mats)
        if K == 1:
            s4 = 1.0
        else:
            agreements = []
            for i in range(K):
                for j in range(i + 1, K):
                    a, b = mats[i].astype(bool), mats[j].astype(bool)
                    inter = (a & b).sum()
                    union = (a | b).sum()
                    agreements.append(inter / (union + 1e-10))
            s4 = float(np.mean(agreements))

        return CausalResult(
            adjacency=agreement,
            binary_adjacency=binary,
            agreement=agreement,
            method_results=raw,
            s4_score=float(np.clip(s4, 0.0, 1.0)),
        )
