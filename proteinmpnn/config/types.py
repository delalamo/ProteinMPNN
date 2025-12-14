"""Configuration types for ProteinMPNN."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class ModelConfig:
    """Configuration for ProteinMPNN model architecture."""
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 48
    dropout: float = 0.1
    augment_eps: float = 0.0
    ca_only: bool = False
    vocab_size: int = 21


@dataclass
class DesignConfig:
    """Configuration for sequence design constraints."""
    chain_id_dict: Optional[Dict] = None
    fixed_positions_dict: Optional[Dict] = None
    omit_AA_dict: Optional[Dict] = None
    tied_positions_dict: Optional[Dict] = None
    pssm_dict: Optional[Dict] = None
    bias_AA_dict: Optional[Dict] = None
    bias_by_res_dict: Optional[Dict] = None

    def has_tied_positions(self) -> bool:
        """Check if tied positions are configured."""
        return self.tied_positions_dict is not None


@dataclass
class InferenceConfig:
    """Configuration for inference/generation."""
    num_seq_per_target: int = 1
    batch_size: int = 1
    sampling_temp: str = "0.1"
    seed: int = 0
    backbone_noise: float = 0.0
    max_length: int = 200000

    # Output options
    save_score: bool = False
    save_probs: bool = False
    score_only: bool = False
    conditional_probs_only: bool = False
    conditional_probs_only_backbone: bool = False
    unconditional_probs_only: bool = False

    # PSSM options
    pssm_multi: float = 0.0
    pssm_threshold: float = 0.0
    pssm_log_odds_flag: bool = False
    pssm_bias_flag: bool = False

    # AA constraints
    omit_AAs: str = 'X'

    @property
    def temperatures(self):
        """Parse temperature string into list of floats."""
        return [float(t) for t in self.sampling_temp.split()]

    @property
    def num_batches(self):
        """Compute number of batches."""
        return self.num_seq_per_target // self.batch_size
