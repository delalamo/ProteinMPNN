"""Featurization for ProteinMPNN."""

from typing import List, Dict, Optional
import itertools
import numpy as np
import torch

from proteinmpnn.data.types import ProteinBatch, ChainInfo


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def featurize_batch(
    batch: List[Dict],
    device: torch.device,
    chain_dict: Optional[Dict] = None,
    fixed_position_dict: Optional[Dict] = None,
    omit_AA_dict: Optional[Dict] = None,
    tied_positions_dict: Optional[Dict] = None,
    pssm_dict: Optional[Dict] = None,
    bias_by_res_dict: Optional[Dict] = None,
    ca_only: bool = False
) -> ProteinBatch:
    """
    Featurize a batch of protein structures.

    Args:
        batch: List of protein dictionaries
        device: Torch device
        chain_dict: Dictionary mapping protein names to (designed, fixed) chain lists
        fixed_position_dict: Dictionary of fixed positions per chain
        omit_AA_dict: Dictionary of amino acids to omit per position
        tied_positions_dict: Dictionary of tied positions
        pssm_dict: Dictionary of PSSM data
        bias_by_res_dict: Dictionary of per-residue bias
        ca_only: Whether to use CA-only mode

    Returns:
        ProteinBatch containing all featurized data
    """
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])

    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])

    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32)
    pssm_coef_all = np.zeros([B, L_max], dtype=np.float32)
    pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32)
    pssm_log_odds_all = 10000.0 * np.ones([B, L_max, 21], dtype=np.float32)
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32)
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(ALPHABET)], dtype=np.int32)

    chain_info_list = []
    tied_pos_list_of_lists_list = []

    # Determine chain ordering for first batch element
    all_chains = []
    for i, b in enumerate(batch):
        if chain_dict is not None:
            masked_chains, visible_chains = chain_dict[b['name']]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == 'seq_chain_']
            visible_chains = []
        masked_chains = sorted(masked_chains)
        visible_chains = sorted(visible_chains)
        all_chains = masked_chains + visible_chains

    for i, b in enumerate(batch):
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0

        if chain_dict is not None:
            masked_chains, visible_chains = chain_dict[b['name']]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == 'seq_chain_']
            visible_chains = []
        masked_chains = sorted(masked_chains)
        visible_chains = sorted(visible_chains)
        all_chains = masked_chains + visible_chains

        for step, letter in enumerate(all_chains):
            chain_seq = b.get(f'seq_chain_{letter}', '')
            if not chain_seq:
                continue

            chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
            chain_length = len(chain_seq)
            global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
            chain_coords = b[f'coords_chain_{letter}']

            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_mask = np.zeros(chain_length)

                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}'])
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack([
                        chain_coords[f'N_chain_{letter}'],
                        chain_coords[f'CA_chain_{letter}'],
                        chain_coords[f'C_chain_{letter}'],
                        chain_coords[f'O_chain_{letter}']
                    ], 1)

                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(chain_length))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask_list.append(np.ones(chain_length))
                omit_AA_mask_list.append(np.zeros([chain_length, len(ALPHABET)], np.int32))
                pssm_coef_list.append(np.zeros(chain_length))
                pssm_bias_list.append(np.zeros([chain_length, 21]))
                pssm_log_odds_list.append(10000.0 * np.ones([chain_length, 21]))
                bias_by_res_list.append(np.zeros([chain_length, 21]))

            elif letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                masked_chain_length_list.append(chain_length)
                chain_mask = np.ones(chain_length)

                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}'])
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack([
                        chain_coords[f'N_chain_{letter}'],
                        chain_coords[f'CA_chain_{letter}'],
                        chain_coords[f'C_chain_{letter}'],
                        chain_coords[f'O_chain_{letter}']
                    ], 1)

                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(chain_length))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1

                # Fixed positions
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict is not None:
                    fixed_pos_list = fixed_position_dict.get(b['name'], {}).get(letter, [])
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)

                # Omit AA mask
                omit_AA_mask_temp = np.zeros([chain_length, len(ALPHABET)], np.int32)
                if omit_AA_dict is not None:
                    chain_omit = omit_AA_dict.get(b['name'], {}).get(letter, [])
                    for item in chain_omit:
                        idx_AA = np.array(item[0]) - 1
                        AA_idx = np.array([
                            np.argwhere(np.array(list(ALPHABET)) == AA)[0][0]
                            for AA in item[1]
                        ]).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        if len(idx_) > 0:
                            omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)

                # PSSM
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict:
                    chain_pssm = pssm_dict.get(b['name'], {}).get(letter)
                    if chain_pssm:
                        pssm_coef = np.array(chain_pssm['pssm_coef'])
                        pssm_bias = np.array(chain_pssm['pssm_bias'])
                        pssm_log_odds = np.array(chain_pssm['pssm_log_odds'])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)

                # Bias by residue
                if bias_by_res_dict:
                    bias_by_res_list.append(
                        np.array(bias_by_res_dict.get(b['name'], {}).get(letter, np.zeros([chain_length, 21])))
                    )
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        # Tied positions
        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict is not None:
            tied_pos_list = tied_positions_dict.get(b['name'], [])
            if tied_pos_list:
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx + v[0][v_count] - 1)
                                tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx + v_ - 1)
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        # Concatenate chain data
        if not x_chain_list:
            continue

        x = np.concatenate(x_chain_list, 0)
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(fixed_position_mask_list, 0)

        pssm_coef_ = np.concatenate(pssm_coef_list, 0)
        pssm_bias_ = np.concatenate(pssm_bias_list, 0)
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list, 0)
        bias_by_res_ = np.concatenate(bias_by_res_list, 0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        m_pos_pad = np.pad(m_pos, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        omit_AA_mask_pad = np.pad(
            np.concatenate(omit_AA_mask_list, 0),
            [[0, L_max - l], [0, 0]],
            'constant',
            constant_values=(0.0,)
        )
        chain_M[i, :] = m_pad
        chain_M_pos[i, :] = m_pos_pad
        omit_AA_mask[i] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        pssm_bias_pad = np.pad(pssm_bias_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        bias_by_res_all[i, :] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([ALPHABET.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

        chain_info_list.append(ChainInfo(
            letter_list=letter_list,
            visible_list=visible_list,
            masked_list=masked_list,
            masked_chain_lengths=masked_chain_length_list
        ))

    # Handle NaN values
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Compute dihedral mask
    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate([phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1)

    # Convert to tensors
    X_out = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    if ca_only:
        X_out = X_out[:, :, 0]

    return ProteinBatch(
        X=X_out,
        S=torch.from_numpy(S).to(dtype=torch.long, device=device),
        mask=torch.from_numpy(mask).to(dtype=torch.float32, device=device),
        lengths=lengths,
        residue_idx=torch.from_numpy(residue_idx).to(dtype=torch.long, device=device),
        chain_M=torch.from_numpy(chain_M).to(dtype=torch.float32, device=device),
        chain_M_pos=torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device),
        chain_encoding_all=torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device),
        omit_AA_mask=torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device),
        dihedral_mask=torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device),
        pssm_coef=torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device),
        pssm_bias=torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device),
        pssm_log_odds=torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32, device=device),
        bias_by_res=torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device),
        tied_beta=torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device),
        chain_info_list=chain_info_list,
        tied_positions=tied_pos_list_of_lists_list
    )
