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

        # Projection layers for global tokens
        # proj_a projects from hidden_dim to 8 * 64 = 512
        # proj_b projects back from 512 to hidden_dim
        self.proj_a = nn.Linear(384, 8 * 64, bias=False)
        self.proj_b = nn.Linear(8 * 64, 384, bias=False)
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

    def _split_local_and_global_tokens(
        self, 
        x: torch.Tensor, 
        delimitations: torch.Tensor, 
        interleaving_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the interleaved tokens back into local tokens and global tokens.

        Args:
            x: Tensor of shape (B, FS + 3*S, H)
            delimitations: Tensor of shape (B, FS) with binary indicators
            interleaving_mask: Tensor of shape (B, FS + 3*S) indicating special tokens

        Returns:
            local_x: Tensor of shape (B, FS, H)
            global_tokens: Tensor of shape (B, S, H)
        """
        B, FS_plus, H = x.size()
        S = delimitations.sum(dim=1).long().max().item()

        # Initialize tensors
        local_x = torch.zeros((B, x.size(1), H), device=x.device, dtype=x.dtype)
        global_tokens = torch.zeros((B, S, H), device=x.device, dtype=x.dtype)

        for b in range(B):
            fs = delimitations[b].sum().item()
            num_insertions = int(fs)
            # Find positions of special tokens
            special_positions = interleaving_mask[b].nonzero(as_tuple=False).squeeze(1).tolist()
            # Extract global tokens
            global_tokens_b = []
            for pos in special_positions:
                # Assuming the order: start, global, end
                global_token = x[b, pos + 1]
                global_tokens_b.append(global_token)
            if len(global_tokens_b) < S:
                # Pad if necessary
                padding = S - len(global_tokens_b)
                if padding > 0:
                    global_tokens_b += [x.new_zeros(H)] * padding
            global_tokens[b, :len(global_tokens_b)] = torch.stack(global_tokens_b[:S], dim=0)
            # Remove special tokens from x to get local tokens
            mask = ~interleaving_mask[b]
            local_x[b] = x[b, mask]

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
            global_tokens: Tensor of shape (B, S, H)
            start_global_embed: Tensor of shape (1, H)
            end_global_embed: Tensor of shape (1, H)

        Returns:
            x: Tensor after applying local and global attention with interleaved global tokens
            global_tokens: Updated global tokens after global attention
        """
        B, FS, H = x.size()
        _, S, _ = global_tokens.size()

        # Project global tokens
        print(f"{x.size()=}") # x.size()=torch.Size([24, 2048, 64])
        print(f"{delimitations.size()=}") # delimitations.size()=torch.Size([24, 2048])
        print(f"{global_tokens.size()=}") # global_tokens.size()=torch.Size([24, 630, 384])
        projected_global = self.proj_a(global_tokens)  # (B, S, 512)
        projected_global = projected_global.view(B, S, 8 * 64)  # Ensure correct reshaping

        # Interleave global tokens into x
        interleaved_x, interleaving_mask = self._interleave_global_tokens(
            x, projected_global, delimitations, start_global_embed, end_global_embed
        )
        # interleaved_x: (B, FS + 3*S, H)
        # interleaving_mask: (B, FS + 3*S)

        # Apply local sliding window attention
        # Assuming that interleaving_mask is used as attn_mask where True indicates masked positions
        # Convert boolean mask to float mask where True means mask (1.0) and False means unmask (0.0)
        attn_mask = interleaving_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, FS + 3*S)
        attn_mask = attn_mask.float()

        # Pass through the local transformer block
        interleaved_x = self.local_transformer_block(interleaved_x, attn_mask=attn_mask)  # (B, FS + 3*S, H)

        # Split tokens back into local and global
        local_x, updated_global_tokens = self._split_local_and_global_tokens(
            interleaved_x, delimitations, interleaving_mask
        )
        # local_x: (B, FS, H)
        # updated_global_tokens: (B, S, H)

        # Project the updated global tokens back to hidden_dim
        projected_updated_global = self.proj_b(updated_global_tokens)  # (B, S, H)

        # Apply global transformer to the global tokens
        # Assuming causal attention is handled within the global_transformer
        projected_updated_global = self.global_transformer(projected_updated_global)  # (B, S, H)

        # Optionally, re-integrate the updated global tokens back into the local_x
        # This step depends on the model's architecture and desired behavior
        # For this example, we'll return both local and global tokens separately

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
