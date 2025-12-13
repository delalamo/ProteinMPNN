"""Positional and other embeddings for ProteinMPNN."""

import torch
import torch.nn as nn


class PositionalEncodings(nn.Module):
    """Positional encodings based on relative sequence positions."""

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        """
        Initialize positional encodings.

        Args:
            num_embeddings: Output embedding dimension
            max_relative_feature: Maximum relative position to encode
        """
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute positional encodings.

        Args:
            offset: Relative position offsets [B, L, K]
            mask: Chain mask (1 for same chain, 0 for different) [B, L, K]

        Returns:
            Positional embeddings [B, L, K, num_embeddings]
        """
        d = torch.clip(
            offset + self.max_relative_feature,
            0,
            2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)

        d_onehot = torch.nn.functional.one_hot(
            d,
            2 * self.max_relative_feature + 1 + 1
        )
        E = self.linear(d_onehot.float())
        return E
