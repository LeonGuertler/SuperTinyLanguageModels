import torch
import torch.nn as nn
from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from typing import Optional

from models.components.transformer_blocks import GenericTransformerBlock


import torch
import torch.nn as nn
from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from typing import Optional, Tuple

from models.components.transformer_blocks import GenericTransformerBlock

class DualTransformerBlock(nn.Module):
    """
    First apply local sliding window attention + FFN,
    then collect the global tokens, wrap them with start and end tokens,
    and apply causal attention + FFN.
    """
    def __init__(self, model_cfg, hidden_dim, context_window, ffn_cfg, attn_cfg):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window

        self.num_local_tokens_per_global = 8  # Number of local tokens per global token

        # Projection layers for global tokens
        # proj_a projects from 384 to num_local_tokens_per_global * hidden_dim
        # proj_b projects back from num_local_tokens_per_global * hidden_dim to 384
        self.proj_a = nn.Linear(384, self.num_local_tokens_per_global * hidden_dim, bias=False)
        self.proj_b = nn.Linear(self.num_local_tokens_per_global * hidden_dim, 384, bias=False)
        # Share weights between proj_a and proj_b
        self.proj_b.weight = nn.Parameter(self.proj_a.weight.T)

        # Local Transformer Block with Sliding Window Attention
        self.local_transformer_block = GenericTransformerBlock(
            hidden_dim=model_cfg["hidden_dim"],
            context_window=model_cfg["context_window"],
            ffn_cfg=model_cfg["ffn"],
            attn_cfg=model_cfg["attn"],
            depth=None  # Assuming depth is handled inside GenericTransformerBlock
        )

        # Global Transformer Block with Causal Attention
        self.global_transformer = GenericTransformerBlock(
            hidden_dim=384,  # Assuming 8 * 48 = 384 (based on proj_a output)
            context_window=model_cfg["context_window"],
            ffn_cfg={
                "name": "generic",
                "normalization": "rms_norm",
                "params": {
                    "ffn_dim": 256,
                    "activation": "gelu",
                    "normalization": "rms_norm",
                    "bias": True,
                    "dropout": 0.0,
                }
            },
            attn_cfg={
                "name": "rope_attention",
                "normalization": "rms_norm",
                "params": {
                    "num_kv_heads": 4,
                    "num_q_heads": 8,
                    "normalization": "rms_norm",
                    "bias": True, 
                    "dropout": 0.0,
                }
            },
            depth=None  # Assuming depth is handled inside GenericTransformerBlock
        )

    def _interleave_global_tokens(
        self, 
        x: torch.Tensor, 
        global_tokens: torch.Tensor, 
        delimitations: torch.Tensor, 
        start_global_embed: torch.Tensor, 
        end_global_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, FS, H = x.size()
        _, max_global, N, _ = global_tokens.size()  # N is num_local_tokens_per_global

        num_insertions = delimitations.sum(dim=1).long()  # (B,)
        insert_size_per_global = self.num_local_tokens_per_global + 2  # N + 2
        new_lengths = FS + insert_size_per_global * num_insertions  # (B,)
        max_new_length = new_lengths.max().item()

        interleaved_x = torch.zeros((B, max_new_length, H), device=x.device, dtype=x.dtype)
        interleaving_mask = torch.zeros((B, max_new_length), dtype=torch.bool, device=x.device)

        cum_insertions = delimitations.cumsum(dim=1)  # (B, FS)
        cum_insertions_shifted = cum_insertions - delimitations  # Shifted cumulative insertions
        offset = insert_size_per_global * cum_insertions_shifted  # (B, FS)
        new_positions = torch.arange(FS, device=x.device).unsqueeze(0).expand(B, FS) + offset

        interleaved_x.scatter_(1, new_positions.unsqueeze(-1).expand(-1, -1, H), x)

        insert_after = delimitations.bool()  # (B, FS)
        insert_indices = insert_after.nonzero(as_tuple=False)  # Shape: (N_insertions, 2)

        if insert_indices.numel() > 0:
            batch_ids = insert_indices[:, 0]
            token_ids = insert_indices[:, 1]

            start_positions = new_positions[batch_ids, token_ids] + 1
            global_positions = start_positions + 1
            end_positions = global_positions + self.num_local_tokens_per_global

            # Ensure positions do not exceed max_new_length - 1
            end_positions = torch.clamp(end_positions, max=max_new_length - 1)

            interleaved_x[batch_ids, start_positions] = start_global_embed

            insertion_order = delimitations[batch_ids, :].cumsum(dim=1)[range(len(token_ids)), token_ids] - 1
            insertion_order = torch.clamp(insertion_order, max=max_global - 1)

            global_tokens_to_insert = global_tokens[batch_ids, insertion_order]  # (N_insertions, N, H)

            for i in range(self.num_local_tokens_per_global):
                positions = global_positions + i
                interleaved_x[batch_ids, positions] = global_tokens_to_insert[:, i, :]

            interleaved_x[batch_ids, end_positions] = end_global_embed

            interleaving_mask[batch_ids, start_positions] = True
            for i in range(self.num_local_tokens_per_global):
                positions = global_positions + i
                interleaving_mask[batch_ids, positions] = True
            interleaving_mask[batch_ids, end_positions] = True

        return interleaved_x, interleaving_mask



    def _split_local_and_global_tokens(
        self, 
        x: torch.Tensor, 
        delimitations: torch.Tensor, 
        interleaving_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the interleaved tokens back into local tokens and global tokens.

        Args:
            x: Tensor of shape (B, FS + (N+2)*S, H)
            delimitations: Tensor of shape (B, FS) with binary indicators
            interleaving_mask: Tensor of shape (B, FS + (N+2)*S) indicating special tokens

        Returns:
            local_x: Tensor of shape (B, FS, H)
            global_tokens: Tensor of shape (B, S, N, H)
        """
        B, total_length, H = x.size()
        N = self.num_local_tokens_per_global  # Number of tokens per global token

        # Initialize lists to collect local and global tokens
        local_tokens_list = []
        global_tokens_list = []

        for b in range(B):
            interleaving_mask_b = interleaving_mask[b]  # (total_length,)
            x_b = x[b]  # (total_length, H)
            delimitations_b = delimitations[b]  # (FS,)

            # Identify positions of special tokens
            special_positions = interleaving_mask_b.nonzero(as_tuple=False).squeeze(1).tolist()

            # Initialize indices
            idx = 0
            local_tokens_b = []
            global_tokens_b = []

            while idx < total_length:
                if interleaving_mask_b[idx]:
                    # Start token
                    idx += 1
                    # Global tokens
                    global_token_sequence = x_b[idx:idx + N]  # (N, H)
                    global_tokens_b.append(global_token_sequence)
                    idx += N
                    # End token
                    idx += 1
                else:
                    # Local token
                    local_tokens_b.append(x_b[idx])
                    idx += 1

            # Stack local tokens
            local_tokens_b = torch.stack(local_tokens_b, dim=0)  # (FS, H)
            local_tokens_list.append(local_tokens_b)

            # Stack global tokens
            if global_tokens_b:
                global_tokens_b = torch.stack(global_tokens_b, dim=0)  # (S, N, H)
            else:
                global_tokens_b = torch.zeros((0, N, H), device=x.device, dtype=x.dtype)
            global_tokens_list.append(global_tokens_b)

        # Pad local tokens to the same length
        local_lengths = [lt.size(0) for lt in local_tokens_list]
        max_local_length = max(local_lengths)
        local_x = torch.zeros((B, max_local_length, H), device=x.device, dtype=x.dtype)
        for b, lt in enumerate(local_tokens_list):
            local_x[b, :lt.size(0)] = lt

        # Pad global tokens to the same length
        global_lengths = [gt.size(0) for gt in global_tokens_list]
        max_global_length = max(global_lengths)
        global_tokens = torch.zeros((B, max_global_length, N, H), device=x.device, dtype=x.dtype)
        for b, gt in enumerate(global_tokens_list):
            global_tokens[b, :gt.size(0)] = gt

        return local_x, global_tokens


    def forward(
        self, 
        x: torch.Tensor, 
        delimitations: torch.Tensor, 
        global_tokens: torch.Tensor, 
        start_global_embed: torch.Tensor, 
        end_global_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DualTransformerBlock.

        Args:
            x: Tensor of shape (B, FS, H)
            delimitations: Tensor of shape (B, FS) with binary indicators
            global_tokens: Tensor of shape (B, S, 384)
            start_global_embed: Tensor of shape (1, H)
            end_global_embed: Tensor of shape (1, H)

        Returns:
            x: Tensor after applying local and global attention with interleaved global tokens
            global_tokens: Updated global tokens after global attention
        """
        B, FS, H = x.size()
        _, S, _ = global_tokens.size()

        # Project global tokens to multiple local tokens
        projected_global = self.proj_a(global_tokens)  # (B, S, N*H)
        projected_global = projected_global.view(B, S, self.num_local_tokens_per_global, H)  # (B, S, N, H)

        # Interleave global tokens into x
        interleaved_x, interleaving_mask = self._interleave_global_tokens(
            x, projected_global, delimitations, start_global_embed, end_global_embed
        )
        # interleaved_x: (B, FS + (N+2)*S, H)
        # interleaving_mask: (B, FS + (N+2)*S)

        # Apply local sliding window attention
        # Assuming that interleaving_mask is used as attn_mask where True indicates masked positions
        # Convert boolean mask to float mask where True means mask (1.0) and False means unmask (0.0)
        attn_mask = interleaving_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        attn_mask = attn_mask.float()

        # Pass through the local transformer block
        interleaved_x = self.local_transformer_block(interleaved_x, attn_mask=attn_mask)  # (B, L, H)

        # Split tokens back into local and global
        local_x, updated_global_tokens = self._split_local_and_global_tokens(
            interleaved_x, delimitations, interleaving_mask
        )
        # local_x: (B, FS, H)
        # updated_global_tokens: (B, S, N, H)

        # Flatten the updated global tokens before projecting back
        B, S, N, H = updated_global_tokens.size()
        updated_global_tokens_flat = updated_global_tokens.view(B, S, N * H)  # (B, S, N*H)

        # Project the updated global tokens back to 384
        projected_updated_global = self.proj_b(updated_global_tokens_flat)  # (B, S, 384)

        # Apply global transformer to the global tokens
        # Assuming causal attention is handled within the global_transformer
        projected_updated_global = self.global_transformer(projected_updated_global)  # (B, S, 384)

        return local_x, projected_updated_global




class DualCoreModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # Build the transformer
        self.transformer = nn.ModuleDict(
            {
                "drop": nn.Dropout(model_cfg.get("dropout_rate", 0.1)),
                "h": nn.ModuleList(
                    [
                        DualTransformerBlock(
                            model_cfg=model_cfg,
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            ffn_cfg=model_cfg["ffn"],
                            attn_cfg=model_cfg["attn"],
                        )
                        for _ in range(model_cfg["num_layers"])
                    ]
                ),
            }
        )

        # Initialize global_token with correct dimensions
        self.global_token = nn.Parameter(torch.zeros(1, 384))
        self.global_token.requires_grad = True

    def set_embedder(self, embedder):
        """
        Set the embedder for the model.

        Args:
            embedder: The embedder object containing token embeddings.
        """
        self.embedder = embedder

    def forward(self, x, delimitations):
        """
        Forward pass of the DualCoreModel.

        Args:
            x: Tensor of shape (B, FS, H)
            delimitations: Tensor of shape (B, FS) with binary indicators

        Returns:
            Tensor after passing through transformer blocks
        """
        # Apply dropout
        x = self.transformer["drop"](x)

        # Batch size and hidden dimension
        B, FS, H = x.size()

        # Embed special tokens externally
        # Assuming embedder.tokenizer.start_global and end_global are token IDs
        start_global_embed = self.embedder.token_embedder(
            torch.tensor(self.embedder.tokenizer.start_global, device=x.device)
        ).unsqueeze(0)  # Shape: (1, H)
        end_global_embed = self.embedder.token_embedder(
            torch.tensor(self.embedder.tokenizer.end_global, device=x.device)
        ).unsqueeze(0)    # Shape: (1, H)

        # Number of global tokens per batch item
        num_global = delimitations.sum(dim=1).long()  # (B,)

        max_global = num_global.max().item()  # Maximum number of global tokens

        if max_global > 0:
            # Expand and pad global tokens
            global_tokens = self.global_token.expand(B, max_global, -1).clone()  # (B, max_global, 384)

            # Mask global tokens beyond the actual number of insertions
            mask = torch.arange(max_global, device=x.device).unsqueeze(0) < num_global.unsqueeze(1)  # (B, max_global)
            global_tokens = global_tokens * mask.unsqueeze(-1)  # Zero out the unused global tokens
        else:
            global_tokens = torch.zeros((B, 0, 384), device=x.device)  # Handle cases with no global tokens

        # Pass through the transformer blocks
        for block in self.transformer["h"]:
            x, global_tokens = block(
                x,
                delimitations,
                global_tokens,
                start_global_embed,
                end_global_embed
            )

        return x
