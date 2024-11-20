import torch

from models.components.attention import build_attention
from models.components.feedforward import build_ffn

from typing import Optional


class DualTransformerBlock(torch.nn.Module):
    """
    First apply local sliding window attention + ffn,
    then collect the global tokens and apply
    causal attention + ffn
    """
    def __init__(self, hidden_dim, context_window, ffn_cfg, attn_cfg):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.ffn = build_ffn(ffn_cfg)
        self.attn = build_attention(attn_cfg)


        self.proj_a = torch.nn.Linear(384, 8*64, bias=False)
        self.proj_b = torch.nn.Linear(8*64, 384, bias=False)
        # share weights 
        self.proj_a.weight = self.proj_b.weight.T



    

    def _interleave_global_tokens(self, x, global_tokens, delimitations):
        B, FS, H = x.size()
        _, max_global, _ = global_tokens.size()

        # Compute the number of insertions per sequence
        num_insertions = delimitations.sum(dim=1)  # (B,)

        # Ensure max_global >= max(num_insertions)
        assert max_global >= num_insertions.max(), "max_global must be >= max number of insertions"

        # Create masks for insertion
        delimitations = delimitations.bool()  # (B, FS)

        # Compute the indices where global tokens will be inserted
        # For each batch, compute the new indices after insertions
        # This is a bit involved; here's an efficient way using cumulative sums

        # Compute the cumulative number of insertions up to each position
        cum_insertions = delimitations.cumsum(dim=1)  # (B, FS)

        # The new position for each original token
        new_positions = torch.arange(FS, device=x.device).unsqueeze(0).expand(B, FS) + cum_insertions  # (B, FS)

        # Total new length per batch
        new_lengths = FS + num_insertions  # (B,)

        # Determine the maximum new length
        max_new_length = new_lengths.max()

        # Initialize the interleaved tensor
        interleaved_x = x.new_zeros((B, max_new_length, H))

        # Initialize a mask to indicate where global tokens are inserted
        interleaving_mask = torch.zeros((B, max_new_length), dtype=torch.bool, device=x.device)

        # Scatter the original tokens into their new positions
        interleaved_x.scatter_(1, new_positions.unsqueeze(-1).expand(-1, -1, H), x)

        # Now, insert global tokens
        # For each batch, find the positions to insert
        # positions to insert are after each original token where delimitations==1
        # Compute insertion positions
        insertion_positions = new_positions + 1  # Insert after the original token

        # Flatten batch and insertions for indexing
        insertion_positions = insertion_positions.masked_select(delimitations)  # (Total_insertions,)

        # Similarly, flatten global_tokens
        valid_global_tokens = global_tokens[:, :max_insertions := max(num_insertions)].contiguous()
        # Shape: (B, max_insertions, H)

        # To handle varying numbers of insertions, we'll need to map each global token to its insertion position
        # However, PyTorch doesn't support dynamic indexing easily. Instead, iterate over the batch dimension.

        for b in range(B):
            for s in range(num_insertions[b]):
                pos = insertion_positions[b * max_insertions + s].item()
                interleaved_x[b, pos] = valid_global_tokens[b, s]
                interleaving_mask[b, pos] = True

        return interleaved_x, interleaving_mask


        def _local_sliding_window_transformer(self, x):
        pass 


    def forward(self, x, global_tokens, delimitations):
        # Project global tokens
        projected_global = self.proj_a(global_tokens)  # (B, S, 8*64)
        projected_global = projected_global.view(x.size(0), -1, 64)  # Adjust based on intended projection

        # Interleave global tokens into x
        interleaved_x, interleaving_mask = self._interleave_global_tokens(x, projected_global, delimitations)

        # Apply local sliding window attention
        local_attended = self.attn(interleaved_x)  # Implement your local attention mechanism

        # Apply feedforward network
        ffn_output = self.ffn(local_attended)

        # Optionally, handle global tokens with causal attention
        # For example, you might separate global tokens and apply different attention

        return ffn_output
        


    







class DualCoreModel(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # build the transformer
        self.transformer = torch.nn.ModuleDict(
            {
                "drop": torch.nn.Dropout(),
                "h": torch.nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            ffn_cfg=model_cfg["ffn"],
                            attn_cfg=model_cfg["attn"],
                            depth=i
                        )
                        for i in range(model_cfg["num_layers"])
                    ]
                ),
            }
        )

        self.global_token = troch.nn.Parameter(torch.zeros((1, 384)))
        self.global_token.requires_grad = True 

    def set_embedder(self, embedder):
        self.embedder = embedder


    def forward(self, x, delimitations):
        # apply dropout
        x = self.transformer.drop(x)


        # Assuming embedder.tokenizer.start_global and end_global are token IDs
        start_global_embed = self.embedder.token_embedder(
            self.embedder.tokenizer.start_global
        ).unsqueeze(0)  # Shape: (1, H)
        end_global_embed = self.embedder.token_embedder(
            self.embedder.tokenizer.end_global
        ).unsqueeze(0)    # Shape: (1, H)

        # Batch size
        batch_size = x.size(0)

        # initialize as many global tokens as there are delimitations
        # delimitations (B, ?, 1)
        num_global = torch.sum(delimitations, axis=1)  # (B, 1)
        max_global = torch.max(num_global).item()  # max number of global tokens across batch

        # Repeat global token to match the maximum global token count in the batch
        global_tokens = self.global_token.expand(batch_size, max_global, -1)  # (B, max_global, 384)

        # Mask global tokens based on delimitations
        for i, num in enumerate(num_global):
            global_tokens[i, num:] = 0  # Mask tokens beyond the actual number of delimitations for each batch

        # pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x, delimitations, global_tokens)

        return x
