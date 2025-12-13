"""ProteinMPNN neural network model."""

from typing import Optional, Dict, List
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinmpnn.core.layers import EncLayer, DecLayer
from proteinmpnn.core.features import ProteinFeatures, CA_ProteinFeatures
from proteinmpnn.utils.tensor_ops import gather_nodes, cat_neighbors_nodes
from proteinmpnn.data.types import SamplingConfig


class ProteinMPNN(nn.Module):
    """
    ProteinMPNN model for protein sequence design.

    This model uses a graph neural network architecture with:
    - Structural feature extraction from protein backbone
    - Message passing encoder layers
    - Autoregressive decoder layers
    """

    def __init__(
        self,
        num_letters: int = 21,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.1,
        ca_only: bool = False
    ):
        """
        Initialize ProteinMPNN model.

        Args:
            num_letters: Number of amino acid types (21 including unknown)
            node_features: Node feature dimension
            edge_features: Edge feature dimension
            hidden_dim: Hidden layer dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            vocab: Vocabulary size
            k_neighbors: Number of nearest neighbors in graph
            augment_eps: Backbone noise augmentation epsilon
            dropout: Dropout rate
            ca_only: Whether to use CA-only features
        """
        super(ProteinMPNN, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.ca_only = ca_only

        # Feature extraction
        if ca_only:
            self.features = CA_ProteinFeatures(
                node_features, edge_features,
                top_k=k_neighbors, augment_eps=augment_eps
            )
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(
                node_features, edge_features,
                top_k=k_neighbors, augment_eps=augment_eps
            )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
        use_input_decoding_order: bool = False,
        decoding_order: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for computing log probabilities.

        Args:
            X: Backbone coordinates [B, L, 4, 3] or [B, L, 3] for CA-only
            S: Sequence indices [B, L]
            mask: Valid position mask [B, L]
            chain_M: Design mask [B, L]
            residue_idx: Residue indices [B, L]
            chain_encoding_all: Chain labels [B, L]
            randn: Random noise for decoding order [B, L]
            use_input_decoding_order: Whether to use provided decoding order
            decoding_order: Optional decoding order [B, L]

        Returns:
            Log probabilities [B, L, 21]
        """
        device = X.device

        # Extract features
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder - unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Sequence embeddings
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Encoder embeddings for decoder
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        # Decoding order
        chain_M = chain_M * mask
        if not use_input_decoding_order:
            decoding_order = torch.argsort(
                (chain_M + 0.0001) * (torch.abs(randn))
            )

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        # Decoder
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def sample(
        self,
        X: torch.Tensor,
        randn: torch.Tensor,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        residue_idx: torch.Tensor,
        mask: torch.Tensor,
        config: SamplingConfig
    ) -> Dict[str, torch.Tensor]:
        """
        Sample sequences from the model.

        Args:
            X: Backbone coordinates
            randn: Random noise for decoding order
            S_true: True sequence (for fixed positions)
            chain_mask: Design mask
            chain_encoding_all: Chain labels
            residue_idx: Residue indices
            mask: Valid position mask
            config: Sampling configuration

        Returns:
            Dictionary with 'S', 'probs', and 'decoding_order'
        """
        device = X.device

        # Extract features
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Prepare for decoding
        chain_M_pos = config.chain_M_pos if config.chain_M_pos is not None else torch.ones_like(chain_mask)
        chain_mask = chain_mask * chain_M_pos * mask

        decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]

        constant = torch.tensor(config.omit_AAs_np, device=device) if config.omit_AAs_np is not None else torch.zeros(21, device=device)
        constant_bias = torch.tensor(config.bias_AAs_np, device=device) if config.bias_AAs_np is not None else torch.zeros(21, device=device)
        omit_AA_mask_flag = config.omit_AA_mask is not None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        bias_by_res = config.bias_by_res if config.bias_by_res is not None else torch.zeros((N_batch, N_nodes, 21), device=device)

        for t_ in range(N_nodes):
            t = decoding_order[:, t_]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])
            mask_gathered = torch.gather(mask, 1, t[:, None])
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]

            if (mask_gathered == 0).all():
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw, 1,
                    t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1])
                )
                mask_t = torch.gather(mask, 1, t[:, None])

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(
                        mask_bw, 1,
                        t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1])
                    ) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l + 1].scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t)
                    )

                h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]
                logits = self.W_out(h_V_t) / config.temperature
                probs = F.softmax(
                    logits - constant[None, :] * 1e8 + constant_bias[None, :] / config.temperature + bias_by_res_gathered / config.temperature,
                    dim=-1
                )

                if config.pssm_bias_flag and config.pssm_coef is not None:
                    pssm_coef_gathered = torch.gather(config.pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_gathered = torch.gather(config.pssm_bias, 1, t[:, None, None].repeat(1, 1, config.pssm_bias.shape[-1]))[:, 0]
                    probs = (1 - config.pssm_multi * pssm_coef_gathered[:, None]) * probs + config.pssm_multi * pssm_coef_gathered[:, None] * pssm_bias_gathered

                if config.pssm_log_odds_flag and config.pssm_log_odds_mask is not None:
                    pssm_log_odds_mask_gathered = torch.gather(config.pssm_log_odds_mask, 1, t[:, None, None].repeat(1, 1, config.pssm_log_odds_mask.shape[-1]))[:, 0]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)

                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(config.omit_AA_mask, 1, t[:, None, None].repeat(1, 1, config.omit_AA_mask.shape[-1]))[:, 0]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)

                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21), (chain_mask_gathered[:, :, None] * probs[:, None, :]).float())

            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (S_t * chain_mask_gathered + S_true_gathered * (1.0 - chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)

        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

    def tied_sample(
        self,
        X: torch.Tensor,
        randn: torch.Tensor,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        residue_idx: torch.Tensor,
        mask: torch.Tensor,
        config: SamplingConfig,
        tied_pos: List[List[int]],
        tied_beta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Sample sequences with tied positions (for symmetry).

        Args:
            X: Backbone coordinates
            randn: Random noise for decoding order
            S_true: True sequence (for fixed positions)
            chain_mask: Design mask
            chain_encoding_all: Chain labels
            residue_idx: Residue indices
            mask: Valid position mask
            config: Sampling configuration
            tied_pos: List of tied position groups
            tied_beta: Tied position weights

        Returns:
            Dictionary with 'S', 'probs', and 'decoding_order'
        """
        device = X.device

        # Extract features
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Prepare for decoding
        chain_M_pos = config.chain_M_pos if config.chain_M_pos is not None else torch.ones_like(chain_mask)
        chain_mask = chain_mask * chain_M_pos * mask
        decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))

        # Reorder based on tied positions
        new_decoding_order = []
        for t_dec in list(decoding_order[0].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])

        decoding_order = torch.tensor(
            list(itertools.chain(*new_decoding_order)), device=device
        )[None].repeat(X.shape[0], 1)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]

        constant = torch.tensor(config.omit_AAs_np, device=device) if config.omit_AAs_np is not None else torch.zeros(21, device=device)
        constant_bias = torch.tensor(config.bias_AAs_np, device=device) if config.bias_AAs_np is not None else torch.zeros(21, device=device)
        omit_AA_mask_flag = config.omit_AA_mask is not None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        bias_by_res = config.bias_by_res if config.bias_by_res is not None else torch.zeros((N_batch, N_nodes, 21), device=device)

        for t_list in new_decoding_order:
            logits = 0.0
            done_flag = False

            for t in t_list:
                if (mask[:, t] == 0).all():
                    S_t = S_true[:, t]
                    for t in t_list:
                        h_S[:, t, :] = self.W_s(S_t)
                        S[:, t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:, t:t+1, :]
                    h_E_t = h_E[:, t:t+1, :, :]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:, t:t+1, :, :]
                    mask_t = mask[:, t:t+1]

                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:, t:t+1, :]
                        h_ESV_t = mask_bw[:, t:t+1, :, :] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l + 1][:, t, :] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)

                    h_V_t = h_V_stack[-1][:, t, :]
                    logits += tied_beta[t] * (self.W_out(h_V_t) / config.temperature) / len(t_list)

            if not done_flag:
                bias_by_res_gathered = bias_by_res[:, t, :]
                probs = F.softmax(
                    logits - constant[None, :] * 1e8 + constant_bias[None, :] / config.temperature + bias_by_res_gathered / config.temperature,
                    dim=-1
                )

                if config.pssm_bias_flag and config.pssm_coef is not None:
                    pssm_coef_gathered = config.pssm_coef[:, t]
                    pssm_bias_gathered = config.pssm_bias[:, t]
                    probs = (1 - config.pssm_multi * pssm_coef_gathered[:, None]) * probs + config.pssm_multi * pssm_coef_gathered[:, None] * pssm_bias_gathered

                if config.pssm_log_odds_flag and config.pssm_log_odds_mask is not None:
                    pssm_log_odds_mask_gathered = config.pssm_log_odds_mask[:, t]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)

                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = config.omit_AA_mask[:, t]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)

                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                S_t_repeat = (chain_mask[:, t] * S_t_repeat + (1 - chain_mask[:, t]) * S_true[:, t]).long()

                for t in t_list:
                    h_S[:, t, :] = self.W_s(S_t_repeat)
                    S[:, t] = S_t_repeat
                    all_probs[:, t, :] = probs.float()

        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

    def conditional_probs(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
        backbone_only: bool = False
    ) -> torch.Tensor:
        """
        Compute conditional probabilities p(s_i | s_{-i}, structure).

        Args:
            X: Backbone coordinates
            S: Sequence indices
            mask: Valid position mask
            chain_M: Design mask
            residue_idx: Residue indices
            chain_encoding_all: Chain labels
            randn: Random noise
            backbone_only: If True, compute p(s_i | structure) only

        Returns:
            Log conditional probabilities [B, L, 21]
        """
        device = X.device

        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0, :] == 1)[:, 0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], 21], device=device).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)

            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.0
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.0

            decoding_order = torch.argsort((order_mask[None] + 0.0001) * torch.abs(randn))
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum(
                'ij, biq, bjp->bqp',
                (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]

        return log_conditional_probs

    def unconditional_probs(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute unconditional probabilities p(s_i | structure).

        Args:
            X: Backbone coordinates
            mask: Valid position mask
            residue_idx: Residue indices
            chain_encoding_all: Chain labels

        Returns:
            Log probabilities [B, L, 21]
        """
        device = X.device

        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
