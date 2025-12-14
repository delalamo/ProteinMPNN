"""Transformer layers for ProteinMPNN encoder and decoder."""

import torch
import torch.nn as nn

from proteinmpnn.utils.tensor_ops import cat_neighbors_nodes


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, num_hidden: int, num_ff: int):
        """
        Initialize feed-forward network.

        Args:
            num_hidden: Hidden dimension size
            num_ff: Feed-forward dimension size
        """
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h_V: Input tensor [B, L, C]

        Returns:
            Output tensor [B, L, C]
        """
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class EncLayer(nn.Module):
    """Encoder layer with message passing on protein graph."""

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        num_heads: int = None,
        scale: int = 30
    ):
        """
        Initialize encoder layer.

        Args:
            num_hidden: Hidden dimension size
            num_in: Input dimension size
            dropout: Dropout rate
            num_heads: Number of attention heads (unused, for compatibility)
            scale: Scaling factor for message aggregation
        """
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        mask_V: torch.Tensor = None,
        mask_attend: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass through encoder layer.

        Args:
            h_V: Node hidden states [B, L, C]
            h_E: Edge hidden states [B, L, K, C]
            E_idx: Edge indices [B, L, K]
            mask_V: Node mask [B, L]
            mask_attend: Attention mask [B, L, K]

        Returns:
            Tuple of (updated h_V, updated h_E)
        """
        # Message passing for nodes
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Feed-forward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # Message passing for edges
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E


class DecLayer(nn.Module):
    """Decoder layer with masked autoregressive attention."""

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        num_heads: int = None,
        scale: int = 30
    ):
        """
        Initialize decoder layer.

        Args:
            num_hidden: Hidden dimension size
            num_in: Input dimension size
            dropout: Dropout rate
            num_heads: Number of attention heads (unused, for compatibility)
            scale: Scaling factor for message aggregation
        """
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_V: torch.Tensor = None,
        mask_attend: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            h_V: Node hidden states [B, L, C]
            h_E: Edge hidden states [B, L, K, C]
            mask_V: Node mask [B, L]
            mask_attend: Attention mask [B, L, K]

        Returns:
            Updated node hidden states [B, L, C]
        """
        # Message passing
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        return h_V
