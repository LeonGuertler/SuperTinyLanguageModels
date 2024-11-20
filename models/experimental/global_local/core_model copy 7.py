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
        """Inserts global tokens at delimiter positions while maintaining sequence integrity."""
        batch_size = byte_sequence_emb.size(0)
        assert batch_size == 1, "Not implemented for batch size > 1"
        
        byte_emb = byte_sequence_emb[0]  # [seq_len, hidden_dim]
        delimiter = delimiter_sequence[0]  # [seq_len]
        projected_global = projected_global[0]  # [num_global, hidden_dim]
        
        # Create tensor masks for easier processing
        is_delimiter = delimiter == 1
        num_delimiters = torch.sum(is_delimiter)
        
        # Initialize output containers
        output_parts = []
        global_idx = []
        global_token_count = 0
        total_tokens_processed = 0
        
        # Process sequence in order
        current_sequence = []
        
        for pos in range(byte_sequence_emb.size(1)):
            if not is_delimiter[pos]:
                # Add regular token to current sequence
                current_sequence.append(byte_emb[pos:pos+1])
            else:
                # Add accumulated sequence if it exists
                if current_sequence:
                    sequence_chunk = torch.cat(current_sequence, dim=0)
                    output_parts.append(sequence_chunk)
                    total_tokens_processed += sequence_chunk.size(0)
                    current_sequence = []
                
                # Record position for global token
                global_idx.append(total_tokens_processed)
                
                # Add global token sequence
                global_token = torch.cat([
                    start_global_emb.expand(1, -1),
                    projected_global[global_token_count].view(self.down_proj_mult, self.hidden_dim),
                    end_global_emb.expand(1, -1)
                ], dim=0)
                output_parts.append(global_token)
                
                # Update counters
                total_tokens_processed += 2 + self.down_proj_mult
                global_token_count += 1
        
        # Add final sequence if it exists
        if current_sequence:
            sequence_chunk = torch.cat(current_sequence, dim=0)
            output_parts.append(sequence_chunk)
            total_tokens_processed += sequence_chunk.size(0)
        
        # Combine all parts and add batch dimension
        x_full = torch.cat(output_parts, dim=0).unsqueeze(0)
        global_idx = torch.tensor(global_idx, dtype=torch.long, device=delimiter.device).unsqueeze(0)
        
        # Debug prints
        # print(f"Original sequence length: {byte_sequence_emb.size(1)}")
        # print(f"Number of regular tokens: {byte_sequence_emb.size(1) - num_delimiters}")
        # print(f"Number of delimiters: {num_delimiters}")
        # print(f"Final sequence length: {x_full.size(1)}")
        # print(f"Number of global token positions: {len(global_idx[0])}")
        
        # Verify counts
        assert len(global_idx[0]) == num_delimiters, \
            f"Number of global tokens ({len(global_idx[0])}) != expected ({num_delimiters})"
        assert x_full.size(1) == (byte_sequence_emb.size(1) - num_delimiters + num_delimiters * (2 + self.down_proj_mult)), \
            "Final sequence length mismatch"
        
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

    def _extract_updated_byte_tokens(self, x_full, global_idx):
        """
        Extract the original sequence by removing global tokens and their markers.
        """
        x = x_full[0]  # Remove batch dimension
        global_idx = global_idx[0].tolist()  # Convert to list for easier handling
        
        # Debug info
        # print(f"Extracting tokens from sequence of length {x_full.size(1)}")
        # print(f"Number of global tokens: {len(global_idx)}")
        
        # Store token sequences between global markers
        sequences = []
        current_pos = 0
        
        # Special case: sequence before first global token
        if len(global_idx) > 0:
            if global_idx[0] > 0:  # There are tokens before first global marker
                sequences.append(x[:global_idx[0]])
                current_pos = global_idx[0]
        
        # Process each global token section
        for i, idx in enumerate(global_idx):
            # Skip the section containing global token + markers
            current_pos = idx + 2 + self.down_proj_mult
            
            # If this isn't the last global token, get sequence until next one
            if i < len(global_idx) - 1:
                next_idx = global_idx[i + 1]
                if next_idx > current_pos:  # Only if there are tokens between markers
                    sequences.append(x[current_pos:next_idx])
        
        # Don't forget sequence after last global token
        if current_pos < x.size(0):
            sequences.append(x[current_pos:])
        
        # Debug info for verification
        total_sequence_length = sum(seq.size(0) for seq in sequences)
        # print(f"Reconstructed sequence length: {total_sequence_length}")
        # print(f"Number of sequences: {len(sequences)}")
        if sequences:
            pass
            # print(f"First sequence length: {sequences[0].size(0)}")
            # print(f"Last sequence length: {sequences[-1].size(0)}")
        
        # Combine all sequences
        if sequences:
            result = torch.cat(sequences, dim=0).unsqueeze(0)
        else:
            # Handle edge case where there are no sequences (all tokens were global)
            result = torch.zeros((1, 0, x.size(-1)), device=x.device)
        
        # Calculate expected length
        inserted_tokens = len(global_idx) * (2 + self.down_proj_mult)  # space taken by global tokens
        expected_length = x_full.size(1) - inserted_tokens
        # print(f"Expected sequence length: {expected_length}")
        # print(f"Actual sequence length: {result.size(1)}")
        
        # Verify sequence length
        assert result.size(1) == expected_length, \
            f"Extracted sequence length {result.size(1)} doesn't match expected length {expected_length}"
            
        return result

    def forward(self, x, delimitations):
        # print("Input size:", x.size())
        # print("First 10 delimiters:", delimitations[0][:10])
        # print("Last 10 delimiters:", delimitations[0][-10:])
        
        num_global_tokens = torch.sum(delimitations)
        # print(f"Number of global tokens: {num_global_tokens}")
        
        global_tokens = self.global_token.expand(1, num_global_tokens, -1)
        projected_global = self.proj_a(global_tokens)

        # Get embeddings for special tokens
        start_global_emb = self.embedder(torch.tensor([self.embedder.tokenizer.start_global], device=x.device))
        end_global_emb = self.embedder(torch.tensor([self.embedder.tokenizer.end_global], device=x.device))

        x_full, global_idx = self._insert_global_tokens(
            x, delimitations, projected_global,
            start_global_emb, end_global_emb
        )

        # print("After insertion size:", x_full.size())

        for local_block, global_block in zip(self.byte_transformer.h, self.global_transformer.h):
            mask = self._create_sliding_window_mask(
                seq_length=x_full.size(1),
                window_size=64,
                batch_size=1,
                num_heads=8
            )
            x_full = local_block(x_full, attn_mask=mask)
            x_full = self._update_global_values(x_full, global_idx, global_block)

        # Extract regular tokens (without delimiters and global tokens)
        output_regular = self._extract_updated_byte_tokens(x_full, global_idx)
        # print("Output regular size:", output_regular.size())
        
        # Reconstruct the full output sequence by reinserting delimiters
        batch_size, regular_seq_len, hidden_dim = output_regular.size()
        input_seq_len = x.size(1)
        
        # Initialize the full output tensor
        output_full = torch.zeros((batch_size, input_seq_len, hidden_dim), device=x.device)
        
        # Create a boolean mask where delimitations are delimiters
        delimiter_mask = delimitations.bool()
        
        # Assign the original delimiter tokens to their positions in the output
        output_full[delimiter_mask] = x[delimiter_mask]
        
        # Assign the updated regular tokens to non-delimiter positions
        output_full[~delimiter_mask] = output_regular
        
        # print("Reconstructed output size:", output_full.size())

        # input(output_full)
        
        # Verify that the output sequence length matches the input sequence length
        assert output_full.size(1) == x.size(1), \
            f"Output size {output_full.size(1)} doesn't match input size {x.size(1)}"
        
        return output_full

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
