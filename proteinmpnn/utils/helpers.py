"""Helper functions for sequence manipulation and scoring."""

from typing import List, Tuple
import numpy as np
import torch


# Standard amino acid alphabet
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
ALPHABET_DICT = {aa: i for i, aa in enumerate(ALPHABET)}


def scores_from_log_probs(
    S: torch.Tensor,
    log_probs: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute negative log probability scores.

    Args:
        S: Sequence indices [B, L]
        log_probs: Log probabilities [B, L, 21]
        mask: Valid position mask [B, L]

    Returns:
        Per-sequence scores [B]
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def sequence_to_string(S: torch.Tensor, mask: torch.Tensor) -> str:
    """
    Convert sequence tensor to amino acid string.

    Args:
        S: Sequence indices [L]
        mask: Valid position mask [L]

    Returns:
        Amino acid sequence string
    """
    seq = ''.join([
        ALPHABET[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0
    ])
    return seq


def parse_fasta(
    filename: str,
    limit: int = -1,
    omit: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a FASTA file.

    Args:
        filename: Path to FASTA file
        limit: Maximum number of sequences to read (-1 for unlimited)
        omit: Characters to omit from sequences

    Returns:
        Tuple of (headers, sequences) as numpy arrays
    """
    if omit is None:
        omit = []

    header = []
    sequence = []

    with open(filename, "r") as lines:
        for line in lines:
            line = line.rstrip()
            if line[0] == ">":
                if len(header) == limit:
                    break
                header.append(line[1:])
                sequence.append([])
            else:
                if omit:
                    line = [item for item in line if item not in omit]
                    line = ''.join(line)
                line = ''.join(line)
                sequence[-1].append(line)

    sequence = [''.join(seq) for seq in sequence]
    return np.array(header), np.array(sequence)


def loss_nll(
    S: torch.Tensor,
    log_probs: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute negative log-likelihood loss.

    Args:
        S: Target sequence indices [B, L]
        log_probs: Predicted log probabilities [B, L, 21]
        mask: Valid position mask [B, L]

    Returns:
        Tuple of (per-position loss, average loss)
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_smoothed(
    S: torch.Tensor,
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute label-smoothed negative log-likelihood loss.

    Args:
        S: Target sequence indices [B, L]
        log_probs: Predicted log probabilities [B, L, 21]
        mask: Valid position mask [B, L]
        weight: Label smoothing weight

    Returns:
        Tuple of (per-position loss, average loss)
    """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av
