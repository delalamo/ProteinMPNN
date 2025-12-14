"""Configuration loaders for ProteinMPNN."""

import json
import os
from typing import Dict, Optional
import torch

from proteinmpnn.config.types import DesignConfig, ModelConfig
from proteinmpnn.core.model import ProteinMPNN


def load_jsonl_config(path: str) -> Optional[Dict]:
    """
    Load a JSONL configuration file.

    Args:
        path: Path to JSONL file

    Returns:
        Dictionary with merged configuration, or None if file doesn't exist
    """
    if not path or not os.path.isfile(path):
        return None

    result = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                result.update(json.loads(line))
    return result


def load_design_config(
    chain_id_jsonl: str = '',
    fixed_positions_jsonl: str = '',
    omit_AA_jsonl: str = '',
    tied_positions_jsonl: str = '',
    pssm_jsonl: str = '',
    bias_AA_jsonl: str = '',
    bias_by_res_jsonl: str = '',
    verbose: bool = False
) -> DesignConfig:
    """
    Load all design configuration files into a single DesignConfig object.

    Args:
        chain_id_jsonl: Path to chain ID configuration
        fixed_positions_jsonl: Path to fixed positions configuration
        omit_AA_jsonl: Path to omit AA configuration
        tied_positions_jsonl: Path to tied positions configuration
        pssm_jsonl: Path to PSSM configuration
        bias_AA_jsonl: Path to AA bias configuration
        bias_by_res_jsonl: Path to per-residue bias configuration
        verbose: Whether to print loading messages

    Returns:
        DesignConfig with all loaded configurations
    """
    config = DesignConfig()

    # Load each config file
    config.chain_id_dict = load_jsonl_config(chain_id_jsonl)
    if verbose and config.chain_id_dict is None:
        print('-' * 40)
        print('chain_id_jsonl is NOT loaded')

    config.fixed_positions_dict = load_jsonl_config(fixed_positions_jsonl)
    if verbose and config.fixed_positions_dict is None:
        print('-' * 40)
        print('fixed_positions_jsonl is NOT loaded')

    config.omit_AA_dict = load_jsonl_config(omit_AA_jsonl)
    if verbose and config.omit_AA_dict is None:
        print('-' * 40)
        print('omit_AA_jsonl is NOT loaded')

    config.tied_positions_dict = load_jsonl_config(tied_positions_jsonl)
    if verbose and config.tied_positions_dict is None:
        print('-' * 40)
        print('tied_positions_jsonl is NOT loaded')

    # PSSM is loaded differently (merged into single dict)
    if pssm_jsonl and os.path.isfile(pssm_jsonl):
        config.pssm_dict = {}
        with open(pssm_jsonl, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    config.pssm_dict.update(json.loads(line))
    elif verbose:
        print('-' * 40)
        print('pssm_jsonl is NOT loaded')

    config.bias_AA_dict = load_jsonl_config(bias_AA_jsonl)
    if verbose and config.bias_AA_dict is None:
        print('-' * 40)
        print('bias_AA_jsonl is NOT loaded')

    config.bias_by_res_dict = load_jsonl_config(bias_by_res_jsonl)
    if verbose and config.bias_by_res_dict is None:
        print('-' * 40)
        print('bias by residue dictionary is not loaded, or not provided')
    elif verbose and config.bias_by_res_dict is not None:
        print('bias by residue dictionary is loaded')

    return config


def get_model_path(
    model_name: str = 'v_48_020',
    ca_only: bool = False,
    use_soluble_model: bool = False,
    path_to_model_weights: str = ''
) -> str:
    """
    Get the path to model weights.

    Args:
        model_name: Name of the model version
        ca_only: Whether to use CA-only model
        use_soluble_model: Whether to use soluble protein model
        path_to_model_weights: Custom path to model weights folder

    Returns:
        Path to checkpoint file
    """
    if path_to_model_weights:
        model_folder_path = path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
    else:
        # Use relative path from this file
        file_path = os.path.realpath(__file__)
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

        if ca_only:
            model_folder_path = os.path.join(base_path, 'ca_model_weights/')
        elif use_soluble_model:
            model_folder_path = os.path.join(base_path, 'soluble_model_weights/')
        else:
            model_folder_path = os.path.join(base_path, 'vanilla_model_weights/')

    return os.path.join(model_folder_path, f'{model_name}.pt')


def load_model(
    model_name: str = 'v_48_020',
    ca_only: bool = False,
    use_soluble_model: bool = False,
    path_to_model_weights: str = '',
    device: torch.device = None,
    backbone_noise: float = 0.0
) -> tuple:
    """
    Load a ProteinMPNN model from checkpoint.

    Args:
        model_name: Name of the model version
        ca_only: Whether to use CA-only model
        use_soluble_model: Whether to use soluble protein model
        path_to_model_weights: Custom path to model weights folder
        device: Torch device to load model on
        backbone_noise: Backbone noise augmentation level

    Returns:
        Tuple of (model, checkpoint_info)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_model_path(
        model_name, ca_only, use_soluble_model, path_to_model_weights
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    hidden_dim = 128
    num_layers = 3

    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=backbone_noise,
        k_neighbors=checkpoint['num_edges']
    )

    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    checkpoint_info = {
        'num_edges': checkpoint['num_edges'],
        'noise_level': checkpoint.get('noise_level', 0.0)
    }

    return model, checkpoint_info
