"""
BAT: Bi-Attention Augmented GNN.

Processes a patient's personalized knowledge-graph subgraph through L layers
of bi-attention message-passing:

  - α attention (node-level): learned softmax over visits per node
  - β attention (visit-level decay): exponential recency weighting
  - Edge attention: relation embedding projected to a scalar gate
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


class BiAttentionGNNConv(MessagePassing):
    """Bi-attention message-passing layer with optional edge attention."""

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
        if x[1] is not None:
            out = out + (1 + self.eps) * x[1]
        w_rel = self.W_R(edge_attr) if self.W_R is not None else None
        return self.nn(out), w_rel

    def message(self, x_j: Tensor, edge_attr: Tensor, attn: Tensor) -> Tensor:
        if self.edge_attn:
            w_rel = self.W_R(edge_attr)
            return (x_j * attn + w_rel * edge_attr).relu()
        return (x_j * attn).relu()

    def __repr__(self):
        return f"BiAttentionGNNConv(nn={self.nn})"


class BAT(nn.Module):
    """Bi-Attention Augmented GNN (BAT).

    Args:
        num_nodes (int): Number of cluster nodes.
        num_rels (int): Number of cluster relations.
        max_visit (int): Maximum visits per patient.
        embedding_dim (int): Dimension of pre-trained embeddings.
        hidden_dim (int): GNN hidden dimension.
        out_channels (int): Output classes / labels.
        layers (int): Number of GNN layers.
        dropout (float): Dropout probability.
        decay_rate (float): Temporal decay γ.
        node_emb (Tensor, optional): Pre-trained node embeddings.
        rel_emb (Tensor, optional): Pre-trained relation embeddings.
        freeze (bool): Freeze embeddings.
        patient_mode (str): "joint" | "graph" | "node".
        use_alpha (bool): Enable node-level attention.
        use_beta (bool): Enable visit-decay attention.
        use_edge_attn (bool): Enable edge attention.
        self_attn (float): Self-loop weight ε.
        gnn (str): GNN variant — "BAT" | "GAT" | "GIN".
        attn_init (Tensor, optional): Initial weights for α attention.
        drop_rate (float): Input edge-dropout fraction.
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
        gnn: str = "BAT",
        attn_init=None,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self.gnn = gnn
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
        self.layers_count = layers
        self.dropout = dropout

        j = torch.arange(max_visit).float()
        self.lambda_j = (
            torch.exp(decay_rate * (max_visit - j))
            .unsqueeze(0)
            .reshape(1, max_visit, 1)
            .float()
        )

        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=freeze)

        if rel_emb is None:
            self.rel_emb = nn.Embedding(num_rels, embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=freeze)

        self.lin = nn.Linear(embedding_dim, hidden_dim)

        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()

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

            if gnn == "BAT":
                self.conv[str(layer)] = BiAttentionGNNConv(
                    nn.Linear(hidden_dim, hidden_dim),
                    edge_dim=hidden_dim,
                    edge_attn=use_edge_attn,
                    eps=self_attn,
                )
            elif gnn == "GAT":
                from torch_geometric.nn import GATConv
                self.conv[str(layer)] = GATConv(hidden_dim, hidden_dim)
            elif gnn == "GIN":
                from torch_geometric.nn import GINConv
                self.conv[str(layer)] = GINConv(nn.Linear(hidden_dim, hidden_dim))

        in_dim = hidden_dim * 2 if patient_mode == "joint" else hidden_dim
        self.MLP = nn.Linear(in_dim, out_channels)

    def to(self, device):
        super().to(device)
        self.lambda_j = self.lambda_j.to(device)
        return self

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
            node_ids   : [num_graph_nodes] cluster node IDs
            rel_ids    : [num_edges] cluster relation IDs
            edge_index : [2, num_edges]
            batch      : [num_graph_nodes]
            visit_node : [B, max_visit, num_nodes]
            ehr_nodes  : list of [num_nodes] tensors
        """
        if in_drop and self.drop_rate > 0:
            n = edge_index.size(1)
            perm = torch.randperm(n, device=edge_index.device)[: int(n * (1 - self.drop_rate))]
            edge_index = edge_index[:, perm]
            rel_ids = rel_ids[perm]

        x = self.lin(self.node_emb(node_ids).float())
        edge_attr = self.lin(self.rel_emb(rel_ids).float())

        if store_attn:
            self.alpha_weights, self.beta_weights = [], []
            self.attention_weights, self.edge_weights = [], []

        for layer in range(1, self.layers_count + 1):
            B = batch.max().item() + 1
            device = edge_index.device

            if self.use_alpha:
                alpha = torch.softmax(self.alpha_attn[str(layer)](visit_node.float()), dim=1)
            if self.use_beta:
                beta = torch.tanh(self.beta_attn[str(layer)](visit_node.float())) * self.lambda_j

            if self.use_alpha and self.use_beta:
                attn = alpha * beta
            elif self.use_alpha:
                attn = alpha * torch.ones(B, self.max_visit, 1, device=device)
            elif self.use_beta:
                attn = beta * torch.ones(B, self.max_visit, self.num_nodes, device=device)
            else:
                attn = torch.ones(B, self.max_visit, self.num_nodes, device=device)

            attn = attn.sum(dim=1)
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn = attn[xj_batch, xj_node_ids].reshape(-1, 1)

            if self.gnn == "BAT":
                x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn)
            else:
                x = self.conv[str(layer)](x, edge_index)
                w_rel = None

            x = F.dropout(F.relu(x), p=0.5, training=self.training)

            if store_attn:
                self.alpha_weights.append(alpha if self.use_alpha else None)
                self.beta_weights.append(beta if self.use_beta else None)
                self.attention_weights.append(attn)
                self.edge_weights.append(w_rel)

        if self.patient_mode in ("joint", "graph"):
            x_graph = F.dropout(global_mean_pool(x, batch), p=self.dropout, training=self.training)

        if self.patient_mode in ("joint", "node"):
            B = batch.max().item() + 1
            x_node = F.dropout(
                self.lin(
                    torch.stack([
                        ehr_nodes[i].view(1, -1) @ self.node_emb.weight / ehr_nodes[i].sum()
                        for i in range(B)
                    ]).squeeze(1)
                ),
                p=self.dropout,
                training=self.training,
            )

        if self.patient_mode == "joint":
            logits = self.MLP(F.dropout(torch.cat([x_graph, x_node], dim=1), p=self.dropout, training=self.training))
        elif self.patient_mode == "graph":
            logits = self.MLP(x_graph)
        else:
            logits = self.MLP(x_node)

        if store_attn:
            return logits, self.alpha_weights, self.beta_weights, self.attention_weights, self.edge_weights
        return logits
