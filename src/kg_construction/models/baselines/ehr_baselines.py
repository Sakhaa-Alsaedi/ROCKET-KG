"""
EHR sequence-model baselines (no knowledge graph).

Models included:
  - RNN        — standard GRU-based sequence model
  - Transformer — multi-head self-attention encoder
  - RETAIN     — reverse-time attention model (Choi et al., 2016)
  - MLP        — simple feedforward baseline

All models share the same interface:
  forward(x, lengths) → logits

where ``x`` is a padded 3-D tensor [B, T, input_dim] and ``lengths`` is a
1-D tensor [B] with the true sequence length per patient.

These are intended as *competitive* baselines, not minimal stubs.  For
drug recommendation (multi-label) and other multi-class tasks, wrap the
output in the appropriate loss function.

Note on SafeDrug / MICRON / GAMENet
-------------------------------------
These pharmacy-specific models depend on external molecule graphs and
DDI matrices. They are available via PyHealth (``pyhealth.models``).
Import them directly:

    from pyhealth.models import SafeDrug, MICRON, GAMENet
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RNN baseline
# ---------------------------------------------------------------------------

class RNN(nn.Module):
    """GRU-based EHR sequence model.

    Args:
        input_dim  : Dimension of per-visit feature vector.
        hidden_dim : GRU hidden dimension.
        out_channels : Number of output classes.
        num_layers : Number of GRU layers.
        dropout    : Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x       : [B, T, input_dim]
            lengths : [B] true sequence lengths (used for packing)
        Returns:
            logits [B, out_channels]
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)
        h = self.dropout(h_n[-1])        # last-layer hidden state
        return self.fc(h)


# ---------------------------------------------------------------------------
# Transformer baseline
# ---------------------------------------------------------------------------

class TransformerBaseline(nn.Module):
    """Multi-head self-attention encoder for EHR sequences.

    Args:
        input_dim   : Dimension of per-visit feature vector.
        hidden_dim  : Transformer model dimension (d_model).
        out_channels: Number of output classes.
        num_heads   : Number of attention heads.
        num_layers  : Number of encoder layers.
        dropout     : Dropout probability.
        max_len     : Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 50,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x       : [B, T, input_dim]
            lengths : [B] — used to build a key-padding mask
        Returns:
            logits [B, out_channels]
        """
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.input_proj(x) + self.pos_emb(pos)

        src_key_padding_mask = None
        if lengths is not None:
            # True at positions that should be ignored (padding)
            src_key_padding_mask = (
                torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            )

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        # Mean-pool over valid timesteps
        if lengths is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, T, 1]
            h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.fc(self.dropout(h))


# ---------------------------------------------------------------------------
# RETAIN baseline
# ---------------------------------------------------------------------------

class RETAIN(nn.Module):
    """Reverse-Time Attention Model (RETAIN).

    Choi et al., "RETAIN: An Interpretable Predictive Model for Healthcare
    using Reverse Time Attention Mechanism", NeurIPS 2016.

    Two GRUs produce:
      α_t — visit-level importance weights (softmax over visits)
      β_t — feature-level importance weights (tanh per visit)

    The context vector is c = Σ_t α_t * (β_t ⊙ x_t).

    Args:
        input_dim   : Dimension of per-visit feature vector.
        hidden_dim  : GRU hidden dimension.
        out_channels: Number of output classes.
        dropout     : Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        # GRU-α: visit-level importance
        self.gru_alpha = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.w_alpha = nn.Linear(hidden_dim, 1)

        # GRU-β: feature-level importance
        self.gru_beta = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.w_beta = nn.Linear(hidden_dim, input_dim)

        self.fc = nn.Linear(input_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x       : [B, T, input_dim] — visits in *forward* order
                      (RETAIN internally reverses the sequence)
            lengths : [B] — true sequence lengths
        Returns:
            logits [B, out_channels]
        """
        # Reverse sequence
        x_rev = torch.flip(x, dims=[1])

        h_alpha, _ = self.gru_alpha(x_rev)      # [B, T, hidden]
        h_beta, _ = self.gru_beta(x_rev)         # [B, T, hidden]

        # α: softmax visit importance
        e = self.w_alpha(h_alpha).squeeze(-1)    # [B, T]
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)  # [B, T, 1]

        # β: tanh feature weights
        beta = torch.tanh(self.w_beta(h_beta))   # [B, T, input_dim]

        # Context vector
        context = (alpha * beta * x_rev).sum(dim=1)  # [B, input_dim]
        return self.fc(self.dropout(context))


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple 2-layer MLP that operates on a flattened visit history.

    Args:
        input_dim   : Dimension of the flattened input.
        hidden_dim  : Hidden layer dimension.
        out_channels: Number of output classes.
        dropout     : Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x       : [B, T, input_dim] or [B, input_dim] — visits or flat features
            lengths : ignored (kept for API consistency)
        Returns:
            logits [B, out_channels]
        """
        if x.dim() == 3:
            x = x.mean(dim=1)   # mean-pool over visits
        return self.net(x)
