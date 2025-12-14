"""
ProteinMPNN - Protein sequence design using message passing neural networks.

This package provides a modular implementation of ProteinMPNN for protein
sequence design. The codebase is organized into focused modules:

- core: Neural network model and layers
- data: Data types, parsing, featurization, and datasets
- config: Configuration types and loaders
- inference: Sampling, scoring, and inference runner
- utils: Utility functions for tensor operations
"""

from proteinmpnn.core.model import ProteinMPNN
from proteinmpnn.data.types import (
    ProteinBatch,
    SamplingConfig,
    SampleResult,
)
from proteinmpnn.config.types import DesignConfig, ModelConfig
from proteinmpnn.config.loaders import load_design_config, load_model
from proteinmpnn.inference.runner import ProteinMPNNRunner

__version__ = "1.0.0"
__all__ = [
    "ProteinMPNN",
    "ProteinBatch",
    "SamplingConfig",
    "SampleResult",
    "DesignConfig",
    "ModelConfig",
    "load_design_config",
    "load_model",
    "ProteinMPNNRunner",
]
