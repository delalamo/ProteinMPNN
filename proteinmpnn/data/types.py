"""Data types for ProteinMPNN."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import torch


@dataclass
class ChainInfo:
    """Chain information for output formatting."""
    letter_list: List[str]
    visible_list: List[str]
    masked_list: List[str]
    masked_chain_lengths: List[int]


@dataclass
class ProteinBatch:
    """
    Container for featurized protein batch data.

    This class encapsulates all tensors needed for model inference,
    replacing the previous 20-tuple return from tied_featurize().
    """
    # Core tensors
    X: torch.Tensor                    # Coordinates [B, L, 4, 3] or [B, L, 3]
    S: torch.Tensor                    # Sequence indices [B, L]
    mask: torch.Tensor                 # Valid position mask [B, L]
    lengths: np.ndarray                # Sequence lengths [B]
    residue_idx: torch.Tensor          # Residue indices [B, L]

    # Chain masks
    chain_M: torch.Tensor              # Design mask [B, L]
    chain_M_pos: torch.Tensor          # Position design mask [B, L]
    chain_encoding_all: torch.Tensor   # Chain IDs [B, L]

    # Constraint tensors
    omit_AA_mask: torch.Tensor         # [B, L, 21]
    dihedral_mask: torch.Tensor        # [B, L, 3]

    # PSSM tensors
    pssm_coef: torch.Tensor            # [B, L]
    pssm_bias: torch.Tensor            # [B, L, 21]
    pssm_log_odds: torch.Tensor        # [B, L, 21]
    bias_by_res: torch.Tensor          # [B, L, 21]
    tied_beta: torch.Tensor            # [L]

    # Chain information for output
    chain_info_list: List[ChainInfo]

    # Tied positions
    tied_positions: List[List[List[int]]]


@dataclass
class SamplingConfig:
    """Configuration for sequence sampling."""
    temperature: float = 1.0

    # Global AA constraints
    omit_AAs_np: Optional[np.ndarray] = None
    bias_AAs_np: Optional[np.ndarray] = None

    # Per-position constraints
    chain_M_pos: Optional[torch.Tensor] = None
    omit_AA_mask: Optional[torch.Tensor] = None
    bias_by_res: Optional[torch.Tensor] = None

    # PSSM settings
    pssm_coef: Optional[torch.Tensor] = None
    pssm_bias: Optional[torch.Tensor] = None
    pssm_multi: float = 0.0
    pssm_log_odds_flag: bool = False
    pssm_log_odds_mask: Optional[torch.Tensor] = None
    pssm_bias_flag: bool = False


@dataclass
class SampleResult:
    """Result from sequence sampling."""
    sequence: str                      # Generated amino acid sequence
    score: float                       # Negative log probability score
    global_score: float               # Global score (all positions)
    recovery: float                   # Sequence recovery rate
    temperature: float                # Sampling temperature used
    sample_id: int                    # Sample number


@dataclass
class ProteinStructure:
    """Parsed protein structure data."""
    name: str
    num_chains: int
    sequence: str
    chains: Dict[str, Dict[str, Any]]  # Chain data

    @classmethod
    def from_dict(cls, d: Dict) -> 'ProteinStructure':
        """Create from dictionary (parsed JSONL format)."""
        chains = {}
        for key, value in d.items():
            if key.startswith('seq_chain_'):
                chain_id = key[-1]
                if chain_id not in chains:
                    chains[chain_id] = {}
                chains[chain_id]['sequence'] = value
            elif key.startswith('coords_chain_'):
                chain_id = key[-1]
                if chain_id not in chains:
                    chains[chain_id] = {}
                chains[chain_id]['coords'] = value

        return cls(
            name=d.get('name', ''),
            num_chains=d.get('num_of_chains', len(chains)),
            sequence=d.get('seq', ''),
            chains=chains
        )
