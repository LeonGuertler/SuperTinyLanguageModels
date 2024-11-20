import torch
import torch.nn as nn
from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from typing import Optional

from models.components.transformer_blocks import GenericTransformerBlock
class DualTransformerBlock(nn.Module):
    """
    First apply local sliding window attention + FFN,
    then collect the global tokens, wrap them with start and end tokens,
    and apply causal attention + FFN.
    """
    def __init__(self, hidden_dim, context_window, ffn_cfg, attn_cfg):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.ffn = build_ffn(ffn_cfg)
        self.attn = build_attention(attn_cfg)

        self.proj_a = nn.Linear(hidden_dim, 8 * 64, bias=False)
        self.proj_b = nn.Linear(8 * 64, hidden_dim, bias=False)
        # Share weights
        self.proj_b.weight = self.proj_a.weight.T


        self.local_transformer_block = GenericTransformerBlock(
            hidden_dim=model_cfg["hidden_dim"],
            context_window=model_cfg["context_window"],
            ffn_cfg=model_cfg["ffn"],
            attn_cfg=model_cfg["attn"],
            depth=None
        )



        self.global_transformer = GenericTransformerBlock(
            hidden_dim=384,
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
            depth=None
        )


    def _interleave_global_tokens(self, x, global_tokens, delimitations, start_global_embed, end_global_embed):
        """
        Interleave global tokens into x based on delimitations,
        wrapping each global token with start and end embeddings.

        Args:
            x: Tensor of shape (B, FS, H)
            global_tokens: Tensor of shape (B, S, H)
            delimitations: Tensor of shape (B, FS) with binary indicators
            start_global_embed: Tensor of shape (1, H)
            end_global_embed: Tensor of shape (1, H)

        Returns:
            interleaved_x: Tensor of shape (B, FS + 3*S, H)
            interleaving_mask: Tensor of shape (B, FS + 3*S) indicating special tokens
        """
        B, FS, H = x.size()
        _, max_global, _ = global_tokens.size()

        # Compute the number of insertions per sequence
        num_insertions = delimitations.sum(dim=1).long()  # (B,)

        # Compute new sequence lengths (each insertion adds 3 tokens: start, global, end)
        new_lengths = FS + 3 * num_insertions  # (B,)
        max_new_length = new_lengths.max().item()

        # Initialize the interleaved tensor
        interleaved_x = torch.zeros((B, max_new_length, H), device=x.device, dtype=x.dtype)
        interleaving_mask = torch.zeros((B, max_new_length), dtype=torch.bool, device=x.device)

        # Compute cumulative insertions to determine new positions
        cum_insertions = delimitations.cumsum(dim=1)  # (B, FS)
        new_positions = torch.arange(FS, device=x.device).unsqueeze(0).expand(B, FS) + 3 * cum_insertions

        # Scatter the original tokens into their new positions
        interleaved_x.scatter_(1, new_positions.unsqueeze(-1).expand(-1, -1, H), x)

        # Identify where insertions should occur
        insert_after = delimitations.bool()  # (B, FS)
        insert_indices = insert_after.nonzero(as_tuple=False)  # Shape: (N, 2) where N is total insertions

        if insert_indices.numel() > 0:
            batch_ids = insert_indices[:, 0]  # (N,)
            token_ids = insert_indices[:, 1]  # (N,)

            # Compute insertion positions
            start_positions = new_positions[batch_ids, token_ids] + 1  # Start token
            global_positions = start_positions + 1                   # Global token
            end_positions = global_positions + 1                     # End token

            # Ensure positions do not exceed max_new_length
            end_positions = torch.clamp(end_positions, max=max_new_length - 1)

            # Assign start and end token embeddings
            interleaved_x[batch_ids, start_positions] = start_global_embed
            interleaved_x[batch_ids, end_positions] = end_global_embed

            # Assign global tokens
            # Compute the index of each insertion within its batch
            # This maps each insertion to its corresponding global token
            insertion_order = delimitations[:B, :FS].cumsum(dim=1) - 1  # (B, FS)
            insertion_order = insertion_order[insert_after]  # (N,)

            # Clamp to ensure no index out of range
            insertion_order = torch.clamp(insertion_order, max=max_global - 1)

            # Gather the corresponding global tokens
            global_tokens_to_insert = global_tokens[batch_ids, insertion_order]  # (N, H)

            interleaved_x[batch_ids, global_positions] = global_tokens_to_insert

            # Update the interleaving mask to indicate special tokens
            interleaving_mask[batch_ids, start_positions] = True
            interleaving_mask[batch_ids, global_positions] = True
            interleaving_mask[batch_ids, end_positions] = True

        return interleaved_x, interleaving_mask


    def _split_local_and_global_tokens(self, x, delimitations):


    def forward(self, x, delimitations, global_tokens, start_global_embed, end_global_embed):
        """
        Forward pass of the DualTransformerBlock.

        Args:
            x: Tensor of shape (B, FS, H)
            delimitations: Tensor of shape (B, FS) with binary indicators
            global_tokens: Tensor of shape (B, S, H)
            start_global_embed: Tensor of shape (1, H)
            end_global_embed: Tensor of shape (1, H)

        Returns:
            Tensor after applying local and causal attention with interleaved global tokens
        """
        # Project global tokens
        projected_global = self.proj_a(global_tokens)  # (B, S, 8*64)
        projected_global = projected_global.view(B, -1, 64)  # Adjust based on intended projection

        # Interleave global tokens into x
        interleaved_x, interleaving_mask = self._interleave_global_tokens(
            x, projected_global, delimitations, start_global_embed, end_global_embed
        )

        # apply local sliding window attention
        x = self.local_transformer_block(interleaved_x, attn_mask=interleaving_mask)

        # de-leave the tokens
        x = self._split_local_and_global_tokens(x=x, delimitations=delimitations, interleaving_mask=interleaving_mask)


        # project the global tokens back up

        # apply global transformer to the global tokens


        # return global tokens and local tokens


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
            global_tokens = torch.zeros((B, 0, H), device=x.device)  # Handle cases with no global tokens

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
