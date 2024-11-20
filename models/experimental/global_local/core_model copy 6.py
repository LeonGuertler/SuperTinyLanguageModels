import torch
import torch.nn as nn
from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from typing import Optional, Tuple
from models.components.transformer_blocks import GenericTransformerBlock

class DualCoreModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        
        self.global_token = nn.Parameter(torch.randn(1, model_cfg["global_hidden_dim"]))
        
        # global projections
        self.proj_a = torch.nn.Linear(
            model_cfg["global_hidden_dim"], 
            model_cfg["global_down_proj_mult"]*model_cfg["hidden_dim"],
            bias=False
        )
        self.proj_b = torch.nn.Linear(
            model_cfg["global_down_proj_mult"]*model_cfg["hidden_dim"],
            model_cfg["global_hidden_dim"], 
            bias=False
        )
        
        # share weights
        self.proj_b.weight = torch.nn.Parameter(self.proj_a.weight.T)
        
        self.hidden_dim = model_cfg["hidden_dim"]
        self.down_proj_mult = model_cfg["global_down_proj_mult"]
        
        # initialize transformers
        self.global_transformer = nn.ModuleDict({
            "drop": nn.Dropout(model_cfg.get("dropout_rate", 0.1)),
            "h": nn.ModuleList([
                GenericTransformerBlock(
                    hidden_dim=model_cfg["global_hidden_dim"],
                    context_window=model_cfg["context_window"]//4,
                    ffn_cfg=model_cfg["global_ffn"],
                    attn_cfg=model_cfg["global_attn"],
                ) for _ in range(model_cfg["num_layers"])
            ]),
        })
        
        self.byte_transformer = nn.ModuleDict({
            "drop": nn.Dropout(model_cfg.get("dropout_rate", 0.1)),
            "h": nn.ModuleList([
                GenericTransformerBlock(
                    hidden_dim=model_cfg["hidden_dim"],
                    context_window=model_cfg["context_window"]*4,
                    ffn_cfg=model_cfg["ffn"],
                    attn_cfg=model_cfg["attn"],
                ) for _ in range(model_cfg["num_layers"])
            ]),
        })

    def set_embedder(self, embedder):
        self.embedder = embedder

    def _insert_global_tokens(self, byte_sequence_emb, delimiter_sequence, projected_global, start_global_emb, end_global_emb):
        batch_size = byte_sequence_emb.size(0)
        assert batch_size == 1, "Not implemented for batch size > 1"
        
        byte_emb = byte_sequence_emb[0]
        delimiter = delimiter_sequence[0]
        projected_global = projected_global[0]

        insert_positions = (delimiter == 1).nonzero(as_tuple=False).squeeze()
        if insert_positions.dim() == 0:
            insert_positions = insert_positions.unsqueeze(0)
        insert_positions = insert_positions.tolist()

        split_sizes = [insert_positions[0]] if insert_positions[0] != 0 else []
        for i in range(1, len(insert_positions)):
            split_sizes.append(insert_positions[i] - insert_positions[i-1])
        split_sizes.append(byte_sequence_emb.size(1) - insert_positions[-1])

        splits = torch.split(byte_emb, split_sizes, dim=0)
        start_emb = start_global_emb.expand(1, -1)
        end_emb = end_global_emb.expand(1, -1)

        x_full = []
        global_idx = []
        dim_counter = 0

        for split, p_global in zip(splits, projected_global):
            x_full.append(split)
            dim_counter += split.size(0)
            global_idx.append(dim_counter)
            
            global_tokens = torch.cat([
                start_emb,
                p_global.view(self.down_proj_mult, self.hidden_dim),
                end_emb
            ], dim=0)
            x_full.append(global_tokens)
            dim_counter += 2 + self.down_proj_mult

        x_full = torch.cat(x_full, dim=0).unsqueeze(0)
        global_idx = torch.tensor(global_idx, dtype=torch.long).unsqueeze(0)
        
        return x_full, global_idx

    def _update_global_values(self, x_full, global_idx, global_transformer_block):
        batch_size = x_full.size(0)
        assert batch_size == 1, "Only supports Batch=1"
        
        x = x_full[0]
        global_idx = global_idx[0]
        
        # Extract and process global tokens
        global_tokens = []
        for idx in global_idx:
            global_token = x[idx:idx+self.down_proj_mult].view(-1)
            global_tokens.append(self.proj_b(global_token.unsqueeze(0)))
            
        global_tokens = torch.cat(global_tokens, dim=0).unsqueeze(0)
        global_tokens = global_transformer_block(global_tokens)
        
        # Create a new tensor for the output
        x_new = x.clone()
        
        # Update global token sections
        for i, idx in enumerate(global_idx):
            projected = self.proj_a(global_tokens[0, i])
            x_new[idx+1:idx+self.down_proj_mult+1] = projected.view(self.down_proj_mult, self.hidden_dim)
            
        return x_new.unsqueeze(0)

    # def _extract_updated_byte_tokens(self, x_full, global_idx):
    #     global_idx = global_idx[0]
    #     byte_tokens = []
        
    #     # Add initial sequence
    #     byte_tokens.append(x_full[0, :global_idx[0]])
        
    #     # Add sequences between global tokens
    #     for i in range(1, len(global_idx)):
    #         start = global_idx[i-1] + 2 + self.down_proj_mult
    #         end = global_idx[i]
    #         byte_tokens.append(x_full[0, start:end])
            
    #     # Combine all byte tokens
    #     return torch.cat(byte_tokens, dim=0).unsqueeze(0)


    def _extract_updated_byte_tokens(self, x_full, global_idx):
        global_idx = global_idx[0]
        byte_tokens = []
        
        # Add initial sequence
        byte_tokens.append(x_full[0, :global_idx[0]])
        
        # Add sequences between global tokens
        for i in range(1, len(global_idx)):
            start = global_idx[i-1] + 2 + self.down_proj_mult  # Start after previous global token section
            end = global_idx[i]  # End at next global token
            byte_tokens.append(x_full[0, start:end])
        
        # Add final sequence after last global token
        start = global_idx[-1] + 2 + self.down_proj_mult
        byte_tokens.append(x_full[0, start:])  # Include everything after the last global token
                
        # Combine all byte tokens
        return torch.cat(byte_tokens, dim=0).unsqueeze(0)

    def forward(self, x, delimitations):
        print(x.size())
        print(delimitations[0][:10])
        print(delimitations[0][-10:])
        num_global_tokens = torch.sum(delimitations)
        global_tokens = self.global_token.expand(1, num_global_tokens, -1)
        projected_global = self.proj_a(global_tokens)

        # Get embeddings for special tokens
        start_global_emb = self.embedder(torch.tensor([self.embedder.tokenizer.start_global], device=x.device))
        end_global_emb = self.embedder(torch.tensor([self.embedder.tokenizer.end_global], device=x.device))

        x_full, global_idx = self._insert_global_tokens(
            x, delimitations, projected_global,
            start_global_emb, end_global_emb
        )

        for local_block, global_block in zip(self.byte_transformer.h, self.global_transformer.h):
            # creating sliding window mask
            seq_len = x_full.size(1)
            mask=self._create_sliding_window_mask(
                seq_length=seq_len,
                window_size=64,
                batch_size=1,
                num_heads=8
            )
            # print(x_full.size(), mask.size())
            # input()
            x_full = local_block(
                x_full,
                attn_mask=mask
            )
            x_full = self._update_global_values(x_full, global_idx, global_block)
        x =  self._extract_updated_byte_tokens(x_full, global_idx)
        print(x.size())
        return x

    def _create_sliding_window_mask(self, seq_length, window_size=64, batch_size=1, num_heads=8):
        """
        Creates a sliding window attention mask compatible with scaled_dot_product_attention.
        
        Args:
            seq_length (int): Length of the input sequence
            window_size (int): Size of the attention window (one-sided). Total window is 2*window_size + 1
            batch_size (int): Batch size
            num_heads (int): Number of attention heads
            
        Returns:
            torch.Tensor: Boolean mask of shape (batch_size, num_heads, seq_length, seq_length)
        """
        # Create position indices
        positions = torch.arange(seq_length, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a distance matrix
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Create the window mask
        # Allow attention if the distance is within the window
        mask = distances.abs() <= window_size
        
        # Expand mask for batch size and number of heads
        # [seq_length, seq_length] -> [batch_size, num_heads, seq_length, seq_length]
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, num_heads, seq_length, seq_length)
        
        return mask[0]