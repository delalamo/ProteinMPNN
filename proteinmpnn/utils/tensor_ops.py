"""Tensor operations for graph neural networks."""

import torch


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """
    Gather edge features at neighbor indices.

    Args:
        edges: Edge features [B, N, N, C]
        neighbor_idx: Neighbor indices [B, N, K]

    Returns:
        Neighbor edge features [B, N, K, C]
    """
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """
    Gather node features at neighbor indices.

    Args:
        nodes: Node features [B, N, C]
        neighbor_idx: Neighbor indices [B, N, K]

    Returns:
        Neighbor node features [B, N, K, C]
    """
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """
    Gather node features at neighbor indices (transposed version).

    Args:
        nodes: Node features [B, N, C]
        neighbor_idx: Neighbor indices [B, K]

    Returns:
        Neighbor features [B, K, C]
    """
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(
    h_nodes: torch.Tensor,
    h_neighbors: torch.Tensor,
    E_idx: torch.Tensor
) -> torch.Tensor:
    """
    Concatenate node features with gathered neighbor features.

    Args:
        h_nodes: Node hidden states [B, N, C]
        h_neighbors: Neighbor hidden states [B, N, K, C]
        E_idx: Edge indices [B, N, K]

    Returns:
        Concatenated features [B, N, K, 2C]
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn
