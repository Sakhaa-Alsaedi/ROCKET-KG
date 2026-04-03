"""
Standalone GAT and GIN baselines.

These operate *without* personalized KG attention — they treat the patient
graph as a plain graph and use standard GNN aggregation.  Used as ablation
baselines in the ROCKET benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_mean_pool


class GAT(nn.Module):
    """4-layer Graph Attention Network.

    Args:
        in_channels  : Input feature dimension.
        hidden_channels : Hidden dimension per head.
        out_channels : Output dimension (classes or hidden for downstream MLP).
        heads        : Number of attention heads.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)


class GIN(nn.Module):
    """3-layer Graph Isomorphism Network.

    Args:
        in_channels  : Input feature dimension.
        hidden_channels : Hidden dimension.
        out_channels : Output dimension.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GINConv(nn.Linear(in_channels, hidden_channels))
        self.conv2 = GINConv(nn.Linear(hidden_channels, hidden_channels))
        self.conv3 = GINConv(nn.Linear(hidden_channels, hidden_channels))
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)
