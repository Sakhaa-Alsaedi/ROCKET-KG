"""
CADI: Causal Attention Dual Inference

Dual-path causal reasoning built entirely into the forward pass:

  1. Factual Path       — forward pass on the original patient graph
  2. Counterfactual Path — forward pass on a perturbed graph
                           (node masking + edge dropout + visit ablation)
  3. Causal Contrast    — Δ = factual − counterfactual → per-node causal weight
  4. Causal Attention   — contrast signal amplifies causally important nodes
  5. Dual Fusion Gate   — learnable combination of both paths for final output
"""

import random
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


# ---------------------------------------------------------------------------
# Message-passing layer
# ---------------------------------------------------------------------------

class CADIConv(MessagePassing):
    """Bi-attention message-passing layer used in both CADI paths."""

    def __init__(
        self,
        nn_module: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        edge_attn: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.nn = nn_module
        self.initial_eps = eps
        self.edge_attn = edge_attn
        self.W_R = nn.Linear(edge_dim, 1) if edge_attn else None

        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.W_R is not None:
            self.W_R.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        attn: Tensor = None,
    ) -> tuple:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, attn=attn)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        w_rel = self.W_R(edge_attr) if self.W_R is not None else None
        return self.nn(out), w_rel

    def message(self, x_j: Tensor, edge_attr: Tensor, attn: Tensor) -> Tensor:
        if self.edge_attn:
            w_rel = self.W_R(edge_attr)
            return (x_j * attn + w_rel * edge_attr).relu()
        return (x_j * attn).relu()


# ---------------------------------------------------------------------------
# Counterfactual perturbation module
# ---------------------------------------------------------------------------

class CounterfactualModule(nn.Module):
    """Generates a perturbed (counterfactual) version of the patient graph.

    Three complementary perturbation strategies are applied simultaneously:
      1. Node masking  — drops a fraction of node-feature dimensions
      2. Edge dropout  — removes a fraction of edges
      3. Visit ablation — zeros out a fraction of visits

    All perturbation rates are learnable so the model discovers the optimal
    counterfactual strength end-to-end.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        max_visit: int,
        node_mask_rate: float = 0.15,
        edge_drop_rate: float = 0.15,
        visit_mask_rate: float = 0.10,
    ):
        super().__init__()
        # Learnable perturbation strengths (sigmoid-activated during use)
        self.node_mask_strength = nn.Parameter(torch.tensor(node_mask_rate))
        self.edge_drop_strength = nn.Parameter(torch.tensor(edge_drop_rate))
        self.visit_mask_strength = nn.Parameter(torch.tensor(visit_mask_rate))

    def perturb_nodes(self, x: Tensor) -> Tensor:
        """Drop a Bernoulli-sampled fraction of node-feature dimensions."""
        if not self.training:
            return x
        rate = torch.sigmoid(self.node_mask_strength).item()
        mask = torch.bernoulli(torch.full_like(x[:, :1], 1.0 - rate)).expand_as(x)
        return x * mask

    def perturb_edges(
        self, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple:
        """Randomly drop a fraction of edges."""
        if not self.training:
            return edge_index, edge_attr
        rate = torch.sigmoid(self.edge_drop_strength).item()
        n = edge_index.size(1)
        keep = max(1, int(n * (1.0 - rate)))
        perm = torch.randperm(n, device=edge_index.device)[:keep]
        return edge_index[:, perm], edge_attr[perm]

    def perturb_visits(self, visit_node: Tensor) -> Tensor:
        """Zero out a Bernoulli-sampled fraction of visits."""
        if not self.training:
            return visit_node
        rate = torch.sigmoid(self.visit_mask_strength)
        B, T, _ = visit_node.shape
        mask = torch.bernoulli(
            torch.full((B, T, 1), 1.0 - rate.item(), device=visit_node.device)
        ).expand_as(visit_node)
        return visit_node * mask


# ---------------------------------------------------------------------------
# Causal contrast module
# ---------------------------------------------------------------------------

class CausalContrastModule(nn.Module):
    """Maps Δ = h_factual − h_counter → scalar causal weight per node.

    Nodes with a large causal effect (Δ is large) receive amplified
    representations; causally inert nodes are left unchanged.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_fact: Tensor, x_counter: Tensor, batch: Tensor) -> Tensor:
        """Return [num_nodes, 1] causal weights in (0, 1)."""
        return self.proj(x_fact - x_counter)


# ---------------------------------------------------------------------------
# Dual fusion gate
# ---------------------------------------------------------------------------

class DualFusionGate(nn.Module):
    """Learnable gate that fuses factual and counterfactual graph embeddings.

    output = g * h_fact + (1 − g) * h_counter
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_fact: Tensor, x_counter: Tensor) -> Tensor:
        gate = self.gate_net(torch.cat([x_fact, x_counter], dim=-1))
        return gate * x_fact + (1 - gate) * x_counter


# ---------------------------------------------------------------------------
# Main CADI model
# ---------------------------------------------------------------------------

class CADI(nn.Module):
    """ROCKET-CADI: Causal Attention Dual Inference model.

    Args:
        num_nodes (int): Number of cluster nodes in the KG.
        num_rels (int): Number of cluster relations in the KG.
        max_visit (int): Maximum number of visits per patient.
        embedding_dim (int): Dimension of pre-trained node/relation embeddings.
        hidden_dim (int): Hidden dimension for GNN layers.
        out_channels (int): Number of output classes / labels.
        layers (int): Number of GNN message-passing layers.
        dropout (float): Dropout probability.
        decay_rate (float): Temporal decay rate γ for β attention.
        node_emb (Tensor, optional): Pre-trained node embeddings.
        rel_emb (Tensor, optional): Pre-trained relation embeddings.
        freeze (bool): Freeze pre-trained embeddings if True.
        patient_mode (str): "joint" | "graph" | "node".
        use_alpha (bool): Enable node-level attention.
        use_beta (bool): Enable visit-decay attention.
        use_edge_attn (bool): Enable edge attention.
        self_attn (float): Self-loop weight ε.
        attn_init (Tensor, optional): Initial weights for alpha attention.
        drop_rate (float): Input edge-dropout rate (applied before GNN layers).
        node_mask_rate (float): Initial node-masking rate for counterfactual.
        edge_drop_rate (float): Initial edge-drop rate for counterfactual.
        visit_mask_rate (float): Initial visit-masking rate for counterfactual.
    """

    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        max_visit: int,
        embedding_dim: int,
        hidden_dim: int,
        out_channels: int,
        layers: int = 3,
        dropout: float = 0.5,
        decay_rate: float = 0.03,
        node_emb=None,
        rel_emb=None,
        freeze: bool = False,
        patient_mode: str = "joint",
        use_alpha: bool = True,
        use_beta: bool = True,
        use_edge_attn: bool = True,
        self_attn: float = 0.0,
        attn_init=None,
        drop_rate: float = 0.0,
        node_mask_rate: float = 0.15,
        edge_drop_rate: float = 0.15,
        visit_mask_rate: float = 0.10,
    ):
        super().__init__()

        self.gnn = "CADI"
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.patient_mode = patient_mode
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.edge_attn = use_edge_attn
        self.drop_rate = drop_rate
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.max_visit = max_visit
        self.hidden_dim = hidden_dim
        self.layers_count = layers
        self.dropout = dropout

        # Temporal decay tensor λ_j
        j = torch.arange(max_visit).float()
        self.lambda_j = (
            torch.exp(decay_rate * (max_visit - j))
            .unsqueeze(0)
            .reshape(1, max_visit, 1)
            .float()
        )

        # Embeddings
        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=freeze)

        if rel_emb is None:
            self.rel_emb = nn.Embedding(num_rels, embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=freeze)

        self.lin = nn.Linear(embedding_dim, hidden_dim)

        # Per-layer attention + separate conv for each path + contrast
        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv_fact = nn.ModuleDict()
        self.conv_counter = nn.ModuleDict()
        self.causal_contrast = nn.ModuleDict()

        for layer in range(1, layers + 1):
            if use_alpha:
                self.alpha_attn[str(layer)] = nn.Linear(num_nodes, num_nodes)
                if attn_init is not None:
                    mat = torch.eye(num_nodes).float() * attn_init.float()
                    self.alpha_attn[str(layer)].weight.data.copy_(mat)
                else:
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)].weight)

            if use_beta:
                self.beta_attn[str(layer)] = nn.Linear(num_nodes, 1)
                nn.init.xavier_normal_(self.beta_attn[str(layer)].weight)

            self.conv_fact[str(layer)] = CADIConv(
                nn.Linear(hidden_dim, hidden_dim),
                edge_dim=hidden_dim,
                edge_attn=use_edge_attn,
                eps=self_attn,
            )
            self.conv_counter[str(layer)] = CADIConv(
                nn.Linear(hidden_dim, hidden_dim),
                edge_dim=hidden_dim,
                edge_attn=use_edge_attn,
                eps=self_attn,
            )
            self.causal_contrast[str(layer)] = CausalContrastModule(hidden_dim)

        # Counterfactual perturbation module
        self.counterfactual = CounterfactualModule(
            hidden_dim, num_nodes, max_visit,
            node_mask_rate, edge_drop_rate, visit_mask_rate,
        )
        # Dual fusion gate
        self.fusion = DualFusionGate(hidden_dim)

        # Output MLP
        in_dim = hidden_dim * 2 if patient_mode == "joint" else hidden_dim
        self.MLP = nn.Linear(in_dim, out_channels)

    def to(self, device):
        super().to(device)
        self.lambda_j = self.lambda_j.to(device)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_attention(
        self,
        layer: int,
        visit_node: Tensor,
        edge_index: Tensor,
        node_ids: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Compute collapsed α⊙β attention per edge source node."""
        device = edge_index.device
        B = batch.max().item() + 1

        if self.use_alpha:
            alpha = torch.softmax(
                self.alpha_attn[str(layer)](visit_node.float()), dim=1
            )
        if self.use_beta:
            beta = (
                torch.tanh(self.beta_attn[str(layer)](visit_node.float()))
                * self.lambda_j
            )

        if self.use_alpha and self.use_beta:
            attn = alpha * beta
        elif self.use_alpha:
            attn = alpha * torch.ones(B, self.max_visit, 1, device=device)
        elif self.use_beta:
            attn = beta * torch.ones(B, self.max_visit, self.num_nodes, device=device)
        else:
            attn = torch.ones(B, self.max_visit, self.num_nodes, device=device)

        attn = attn.sum(dim=1)                        # [B, num_nodes]
        xj_node_ids = node_ids[edge_index[0]]
        xj_batch = batch[edge_index[0]]
        return attn[xj_batch, xj_node_ids].reshape(-1, 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        node_ids: Tensor,
        rel_ids: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        visit_node: Tensor,
        ehr_nodes,
        store_attn: bool = False,
        in_drop: bool = False,
    ):
        """
        Args:
            node_ids   : [num_graph_nodes] — cluster node IDs in batch
            rel_ids    : [num_edges] — cluster relation IDs
            edge_index : [2, num_edges] — COO edge connectivity
            batch      : [num_graph_nodes] — batch assignment
            visit_node : [B, max_visit, num_nodes] — visit–node binary matrix
            ehr_nodes  : list of [num_nodes] tensors, one per patient
            store_attn : return intermediate attention weights if True
            in_drop    : apply input edge dropout if True

        Returns:
            logits [B, out_channels], plus optional attention lists
        """
        # --- Optional input edge dropout ---
        if in_drop and self.drop_rate > 0:
            n = edge_index.size(1)
            keep_n = int(n * (1.0 - self.drop_rate))
            perm = torch.randperm(n, device=edge_index.device)[:keep_n]
            edge_index = edge_index[:, perm]
            rel_ids = rel_ids[perm]

        # --- Embeddings ---
        x = self.lin(self.node_emb(node_ids).float())
        edge_attr = self.lin(self.rel_emb(rel_ids).float())

        # --- Counterfactual perturbations ---
        x_cf = self.counterfactual.perturb_nodes(x)
        edge_index_cf, edge_attr_cf = self.counterfactual.perturb_edges(edge_index, edge_attr)
        visit_node_cf = self.counterfactual.perturb_visits(visit_node)

        if store_attn:
            self.attention_weights = []
            self.edge_weights = []
            self.causal_weights = []

        x_fact = x
        x_counter = x_cf

        for layer in range(1, self.layers_count + 1):
            attn_fact = self._compute_attention(layer, visit_node, edge_index, node_ids, batch)
            attn_cf = self._compute_attention(layer, visit_node_cf, edge_index_cf, node_ids, batch)

            x_fact, w_rel_fact = self.conv_fact[str(layer)](x_fact, edge_index, edge_attr, attn=attn_fact)
            x_counter, _ = self.conv_counter[str(layer)](x_counter, edge_index_cf, edge_attr_cf, attn=attn_cf)

            causal_w = self.causal_contrast[str(layer)](x_fact, x_counter, batch)
            x_fact = x_fact * (1.0 + causal_w)

            x_fact = F.dropout(F.relu(x_fact), p=0.5, training=self.training)
            x_counter = F.dropout(F.relu(x_counter), p=0.5, training=self.training)

            if store_attn:
                self.attention_weights.append(attn_fact)
                self.edge_weights.append(w_rel_fact)
                self.causal_weights.append(causal_w)

        # --- Graph-level pooling ---
        if self.patient_mode in ("joint", "graph"):
            x_graph_fact = global_mean_pool(x_fact, batch)
            x_graph_counter = global_mean_pool(x_counter, batch)
            x_graph = F.dropout(
                self.fusion(x_graph_fact, x_graph_counter),
                p=self.dropout,
                training=self.training,
            )

        if self.patient_mode in ("joint", "node"):
            B = batch.max().item() + 1
            x_node = torch.stack([
                ehr_nodes[i].view(1, -1) @ self.node_emb.weight / ehr_nodes[i].sum()
                for i in range(B)
            ])
            x_node = F.dropout(self.lin(x_node).squeeze(1), p=self.dropout, training=self.training)

        if self.patient_mode == "joint":
            logits = self.MLP(F.dropout(torch.cat([x_graph, x_node], dim=1), p=self.dropout, training=self.training))
        elif self.patient_mode == "graph":
            logits = self.MLP(x_graph)
        else:
            logits = self.MLP(x_node)

        if store_attn:
            return logits, self.attention_weights, self.edge_weights, self.causal_weights
        return logits
