"""Main inference runner for ProteinMPNN."""

import os
import copy
import time
import subprocess
from typing import List, Optional, Dict, Any
import numpy as np
import torch

from proteinmpnn.core.model import ProteinMPNN
from proteinmpnn.data.types import ProteinBatch, SamplingConfig, SampleResult
from proteinmpnn.data.parsers import parse_PDB
from proteinmpnn.data.featurize import featurize_batch
from proteinmpnn.data.datasets import StructureDataset, StructureDatasetPDB
from proteinmpnn.config.types import DesignConfig, InferenceConfig
from proteinmpnn.config.loaders import load_model, load_design_config
from proteinmpnn.utils.helpers import scores_from_log_probs, sequence_to_string, parse_fasta


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


class ProteinMPNNRunner:
    """
    High-level interface for ProteinMPNN inference.

    This class provides a simplified API for:
    - Loading models
    - Processing protein structures
    - Generating sequences
    - Scoring sequences
    """

    def __init__(
        self,
        model_name: str = 'v_48_020',
        ca_only: bool = False,
        use_soluble_model: bool = False,
        path_to_model_weights: str = '',
        device: torch.device = None,
        backbone_noise: float = 0.0
    ):
        """
        Initialize the runner with a model.

        Args:
            model_name: Name of the model version
            ca_only: Whether to use CA-only model
            use_soluble_model: Whether to use soluble protein model
            path_to_model_weights: Custom path to model weights
            device: Torch device
            backbone_noise: Backbone noise augmentation
        """
        self.ca_only = ca_only
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model, self.checkpoint_info = load_model(
            model_name=model_name,
            ca_only=ca_only,
            use_soluble_model=use_soluble_model,
            path_to_model_weights=path_to_model_weights,
            device=self.device,
            backbone_noise=backbone_noise
        )

        self.alphabet = ALPHABET
        self.alphabet_dict = {aa: i for i, aa in enumerate(self.alphabet)}

    def load_structure(
        self,
        pdb_path: str = '',
        jsonl_path: str = '',
        chains_to_design: Optional[List[str]] = None,
        max_length: int = 200000
    ) -> tuple:
        """
        Load protein structure from PDB or JSONL file.

        Args:
            pdb_path: Path to PDB file
            jsonl_path: Path to JSONL file with parsed structures
            chains_to_design: List of chains to design
            max_length: Maximum sequence length

        Returns:
            Tuple of (dataset, chain_id_dict)
        """
        chain_id_dict = None

        if pdb_path:
            pdb_dict_list = parse_PDB(pdb_path, ca_only=self.ca_only)
            dataset = StructureDatasetPDB(
                pdb_dict_list, truncate=None, max_length=max_length
            )

            all_chain_list = [
                item[-1:] for item in list(pdb_dict_list[0])
                if item[:9] == 'seq_chain'
            ]

            if chains_to_design:
                designed_chain_list = chains_to_design
            else:
                designed_chain_list = all_chain_list

            fixed_chain_list = [
                letter for letter in all_chain_list
                if letter not in designed_chain_list
            ]

            chain_id_dict = {
                pdb_dict_list[0]['name']: (designed_chain_list, fixed_chain_list)
            }
        else:
            dataset = StructureDataset(
                jsonl_path, truncate=None, max_length=max_length, verbose=False
            )

        return dataset, chain_id_dict

    def featurize(
        self,
        protein: Dict,
        config: DesignConfig,
        chain_id_dict: Optional[Dict] = None
    ) -> ProteinBatch:
        """
        Featurize a protein structure.

        Args:
            protein: Protein dictionary
            config: Design configuration
            chain_id_dict: Optional override for chain ID configuration

        Returns:
            ProteinBatch with featurized data
        """
        batch = [copy.deepcopy(protein)]

        # Use provided chain_id_dict or from config
        effective_chain_dict = chain_id_dict or config.chain_id_dict

        return featurize_batch(
            batch=batch,
            device=self.device,
            chain_dict=effective_chain_dict,
            fixed_position_dict=config.fixed_positions_dict,
            omit_AA_dict=config.omit_AA_dict,
            tied_positions_dict=config.tied_positions_dict,
            pssm_dict=config.pssm_dict,
            bias_by_res_dict=config.bias_by_res_dict,
            ca_only=self.ca_only
        )

    def generate_sequences(
        self,
        batch: ProteinBatch,
        config: DesignConfig,
        num_sequences: int = 1,
        temperature: float = 0.1,
        omit_AAs: str = 'X',
        pssm_multi: float = 0.0,
        pssm_threshold: float = 0.0,
        pssm_log_odds_flag: bool = False,
        pssm_bias_flag: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate sequences for a featurized batch.

        Args:
            batch: Featurized protein batch
            config: Design configuration
            num_sequences: Number of sequences to generate
            temperature: Sampling temperature
            omit_AAs: Amino acids to omit
            pssm_multi: PSSM multiplier
            pssm_threshold: PSSM threshold
            pssm_log_odds_flag: Whether to use PSSM log odds
            pssm_bias_flag: Whether to use PSSM bias

        Returns:
            List of result dictionaries
        """
        results = []

        # Prepare AA constraints
        omit_AAs_np = np.array([AA in omit_AAs for AA in self.alphabet]).astype(np.float32)
        bias_AAs_np = np.zeros(len(self.alphabet))

        if config.bias_AA_dict:
            for n, AA in enumerate(self.alphabet):
                if AA in config.bias_AA_dict:
                    bias_AAs_np[n] = config.bias_AA_dict[AA]

        # PSSM threshold mask
        pssm_log_odds_mask = (batch.pssm_log_odds > pssm_threshold).float()

        # Create sampling config
        sampling_config = SamplingConfig(
            temperature=temperature,
            omit_AAs_np=omit_AAs_np,
            bias_AAs_np=bias_AAs_np,
            chain_M_pos=batch.chain_M_pos,
            omit_AA_mask=batch.omit_AA_mask,
            pssm_coef=batch.pssm_coef,
            pssm_bias=batch.pssm_bias,
            pssm_multi=pssm_multi,
            pssm_log_odds_flag=pssm_log_odds_flag,
            pssm_log_odds_mask=pssm_log_odds_mask,
            pssm_bias_flag=pssm_bias_flag,
            bias_by_res=batch.bias_by_res
        )

        with torch.no_grad():
            for i in range(num_sequences):
                randn = torch.randn(batch.chain_M.shape, device=self.device)

                if config.has_tied_positions() and batch.tied_positions[0]:
                    sample_dict = self.model.tied_sample(
                        X=batch.X,
                        randn=randn,
                        S_true=batch.S,
                        chain_mask=batch.chain_M,
                        chain_encoding_all=batch.chain_encoding_all,
                        residue_idx=batch.residue_idx,
                        mask=batch.mask,
                        config=sampling_config,
                        tied_pos=batch.tied_positions[0],
                        tied_beta=batch.tied_beta
                    )
                else:
                    sample_dict = self.model.sample(
                        X=batch.X,
                        randn=randn,
                        S_true=batch.S,
                        chain_mask=batch.chain_M,
                        chain_encoding_all=batch.chain_encoding_all,
                        residue_idx=batch.residue_idx,
                        mask=batch.mask,
                        config=sampling_config
                    )

                S_sample = sample_dict["S"]

                # Compute scores
                log_probs = self.model(
                    batch.X, S_sample, batch.mask,
                    batch.chain_M * batch.chain_M_pos,
                    batch.residue_idx, batch.chain_encoding_all,
                    randn, use_input_decoding_order=True,
                    decoding_order=sample_dict["decoding_order"]
                )

                mask_for_loss = batch.mask * batch.chain_M * batch.chain_M_pos
                scores = scores_from_log_probs(S_sample, log_probs, mask_for_loss)
                global_scores = scores_from_log_probs(S_sample, log_probs, batch.mask)

                # Compute recovery
                recovery = torch.sum(
                    torch.sum(
                        torch.nn.functional.one_hot(batch.S, 21) *
                        torch.nn.functional.one_hot(S_sample, 21),
                        axis=-1
                    ) * mask_for_loss
                ) / torch.sum(mask_for_loss)

                # Convert to sequence string
                seq = sequence_to_string(S_sample[0], batch.chain_M[0])

                results.append({
                    'sequence': seq,
                    'score': scores[0].item(),
                    'global_score': global_scores[0].item(),
                    'recovery': recovery.item(),
                    'temperature': temperature,
                    'sample_id': i + 1,
                    'S': S_sample,
                    'probs': sample_dict['probs'],
                    'log_probs': log_probs
                })

        return results

    def score_sequence(
        self,
        batch: ProteinBatch,
        sequence: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Score a sequence for the given structure.

        Args:
            batch: Featurized protein batch
            sequence: Optional sequence to score (uses native if not provided)

        Returns:
            Dictionary with scores
        """
        with torch.no_grad():
            S = batch.S.clone()

            if sequence:
                S_input = torch.tensor(
                    [self.alphabet_dict[AA] for AA in sequence],
                    device=self.device
                )[None, :].repeat(batch.X.shape[0], 1)
                S[:, :len(sequence)] = S_input

            randn = torch.randn(batch.chain_M.shape, device=self.device)
            log_probs = self.model(
                batch.X, S, batch.mask,
                batch.chain_M * batch.chain_M_pos,
                batch.residue_idx, batch.chain_encoding_all, randn
            )

            mask_for_loss = batch.mask * batch.chain_M * batch.chain_M_pos
            scores = scores_from_log_probs(S, log_probs, mask_for_loss)
            global_scores = scores_from_log_probs(S, log_probs, batch.mask)

            return {
                'score': scores.cpu().numpy(),
                'global_score': global_scores.cpu().numpy(),
                'sequence': sequence_to_string(S[0], batch.chain_M[0])
            }

    def conditional_probs(
        self,
        batch: ProteinBatch,
        backbone_only: bool = False
    ) -> np.ndarray:
        """
        Compute conditional probabilities.

        Args:
            batch: Featurized protein batch
            backbone_only: If True, compute p(s_i | structure) only

        Returns:
            Log conditional probabilities [B, L, 21]
        """
        with torch.no_grad():
            randn = torch.randn(batch.chain_M.shape, device=self.device)
            log_probs = self.model.conditional_probs(
                batch.X, batch.S, batch.mask,
                batch.chain_M * batch.chain_M_pos,
                batch.residue_idx, batch.chain_encoding_all,
                randn, backbone_only
            )
            return log_probs.cpu().numpy()

    def unconditional_probs(self, batch: ProteinBatch) -> np.ndarray:
        """
        Compute unconditional probabilities.

        Args:
            batch: Featurized protein batch

        Returns:
            Log probabilities [B, L, 21]
        """
        with torch.no_grad():
            log_probs = self.model.unconditional_probs(
                batch.X, batch.mask,
                batch.residue_idx, batch.chain_encoding_all
            )
            return log_probs.cpu().numpy()

    @staticmethod
    def format_sequence_output(
        results: List[Dict],
        chain_info,
        native_sequence: str,
        native_score: float,
        global_native_score: float,
        fixed_chains: List[str],
        designed_chains: List[str],
        model_name: str,
        seed: int,
        ca_only: bool = False
    ) -> str:
        """
        Format results as FASTA output.

        Args:
            results: List of generation results
            chain_info: Chain information
            native_sequence: Native sequence string
            native_score: Native sequence score
            global_native_score: Global native score
            fixed_chains: List of fixed chain IDs
            designed_chains: List of designed chain IDs
            model_name: Name of the model used
            seed: Random seed used
            ca_only: Whether CA-only model was used

        Returns:
            FASTA formatted string
        """
        output_lines = []

        # Get git hash
        try:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            base_dir = os.path.dirname(os.path.dirname(script_dir))
            commit_str = subprocess.check_output(
                f'git --git-dir {base_dir}/.git rev-parse HEAD',
                shell=True, stderr=subprocess.DEVNULL
            ).decode().strip()
        except subprocess.CalledProcessError:
            commit_str = 'unknown'

        print_model_name = 'CA_model_name' if ca_only else 'model_name'

        # Native sequence header
        native_score_str = np.format_float_positional(
            np.float32(native_score), unique=False, precision=4
        )
        global_score_str = np.format_float_positional(
            np.float32(global_native_score), unique=False, precision=4
        )

        output_lines.append(
            f'>{chain_info}, score={native_score_str}, global_score={global_score_str}, '
            f'fixed_chains={fixed_chains}, designed_chains={designed_chains}, '
            f'{print_model_name}={model_name}, git_hash={commit_str}, seed={seed}'
        )
        output_lines.append(native_sequence)

        # Generated sequences
        for result in results:
            score_str = np.format_float_positional(
                np.float32(result['score']), unique=False, precision=4
            )
            global_str = np.format_float_positional(
                np.float32(result['global_score']), unique=False, precision=4
            )
            recovery_str = np.format_float_positional(
                np.float32(result['recovery']), unique=False, precision=4
            )

            output_lines.append(
                f'>T={result["temperature"]}, sample={result["sample_id"]}, '
                f'score={score_str}, global_score={global_str}, seq_recovery={recovery_str}'
            )
            output_lines.append(result['sequence'])

        return '\n'.join(output_lines)
