"""
A collection of embedding models. A collection model includes
the tokenizer(s), token embeddings and positional encodings
(if necessary).
"""

import torch

from models.components.positional_encoding import LearnedPosEncoding
from models.components.tokenizers import build_tokenizer
from models.embedding_models import EmbedderInterface
from models.experimental.byte_level.layers import ByteLevelTransformerBlock

from models.components.transformer_blocks import GenericTransformerBlock
from copy import deepcopy

class TokenizerEncoder(torch.nn.Module):
    """
    Take seq of byte embeddings, return transformed sequence and attention mask.
    """

    def __init__(self, num_delimiter_layers, byte_hidden):
        super().__init__()

        self.num_delimiter_layers = num_delimiter_layers
        self.byte_hidden = byte_hidden

        self.transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=self.byte_hidden,
                    context_window=5 * 2048,
                    ffn_cfg={
                        "ffn_type": "generic",
                        "ffn_dim": 4 * self.byte_hidden,
                        "activation": "gelu",
                        "normalization": "rms_norm",
                        "bias": False,
                        "dropout": 0.0,
                    },
                    attn_cfg={
                        "attn_type": "causal",
                        "num_kv_heads": 8,
                        "num_q_heads": 8,
                        "normalization": "rms_norm",
                        "bias": False,
                        "dropout": False,
                        "pos_enc_cfg": {
                            "positional_encoding_type": "rope"
                        }
                    },
                )
                for _ in range(self.num_delimiter_layers)
            ]
        )

        self.end_of_seq_head = torch.nn.Linear(
            self.byte_hidden,
            1,  # Output logits for delimiter prediction
            bias=True,
        )

    def forward(self, x, x_ids):
        # Pass through transformer blocks
        x_transformed = x
        for block in self.transformer:
            x_transformed = block(x_transformed)

        # Predict delimiters
        logits = self.end_of_seq_head(x_transformed).squeeze(-1)  # Shape: (batch, seq_len)

        # Apply sigmoid activation
        probs = torch.sigmoid(logits)

        # Determine chunk boundaries using a threshold
        threshold = 0.5  # Adjust as needed
        end_of_chunk = probs > threshold  # Shape: (batch, seq_len)

        chunk_len_loss = torch.mean(torch.sum(probs, dim=1))
        

        batch_size, seq_len = end_of_chunk.size()
        device = x.device

        # Initialize lists to store chunk boundaries and average chunk lengths
        chunk_spans = []
        avg_chunk_len = []

        # Set minimum and maximum chunk lengths
        min_chunk_len = 3
        max_chunk_len = 16

        for batch in range(batch_size):
            # Get predicted end positions
            ends = torch.nonzero(end_of_chunk[batch], as_tuple=False).squeeze(-1)

            # If no ends detected, set the end to the last token
            if ends.numel() == 0:
                ends = torch.tensor([seq_len - 1], device=device)
            else:
                ends = ends + 1  # Adjust ends to point after the chunk end

            # Ensure ends are sorted and unique
            ends, _ = torch.sort(ends)

            # Initialize starts
            starts = torch.cat([torch.tensor([0], device=device), ends[:-1]])

            # Adjust chunk boundaries to enforce min and max chunk lengths
            adjusted_starts = []
            adjusted_ends = []

            current_start = 0
            while current_start < seq_len:
                # Set the initial end index
                current_end = min(current_start + max_chunk_len, seq_len)

                # Find the next potential end within the max_chunk_len
                potential_ends = ends[(ends > current_start) & (ends <= current_end)]
                if potential_ends.numel() > 0:
                    current_end = potential_ends[0].item()
                else:
                    current_end = min(current_start + max_chunk_len, seq_len)

                # Enforce minimum chunk length
                if current_end - current_start < min_chunk_len:
                    # Extend the chunk to satisfy min_chunk_len
                    current_end = min(current_start + min_chunk_len, seq_len)

                adjusted_starts.append(current_start)
                adjusted_ends.append(current_end)

                # Move to the next chunk
                current_start = current_end

            starts = torch.tensor(adjusted_starts, device=device)
            ends = torch.tensor(adjusted_ends, device=device)

            chunk_lengths = ends - starts

            # Filter out zero-length chunks (shouldn't occur with the above logic)
            valid = chunk_lengths > 0
            starts = starts[valid]
            ends = ends[valid]

            chunk_spans.append((starts, ends))
            avg_chunk_len.append(chunk_lengths[valid].float().mean())

        avg_chunk_len = sum(avg_chunk_len) / len(avg_chunk_len)

        # Initialize attention masks
        attention_masks = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)

        # Build attention masks based on chunks
        for batch in range(batch_size):
            starts, ends = chunk_spans[batch]
            for start, end in zip(starts.tolist(), ends.tolist()):
                attention_masks[batch, start:end, start:end] = True

        return x_transformed, x_ids, avg_chunk_len, attention_masks, chunk_spans, chunk_len_loss





class ByteBidirectionEncoding(torch.nn.Module):
    """
    Input shape: batch x max_num_chuncks x max_chunck_length x 128 (byte_embed_dim)
    return batch x max_num_chuncks x hidden_dim (512)
    """
    def __init__(self, byte_hidden, hidden_dim, max_chunk_length):
        super().__init__()
        self.max_chunk_length = max_chunk_length 
        self.byte_hidden = byte_hidden
        self.hidden_dim = hidden_dim


        # build the transformer blocks
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=self.byte_hidden,
                    output_dim=self.byte_hidden,
                    ffn_dim=self.byte_hidden*4,
                    context_window=self.max_chunk_length,
                ),
                ByteLevelTransformerBlock(
                    input_dim=self.byte_hidden,
                    output_dim=self.byte_hidden ,
                    ffn_dim=self.byte_hidden*4,
                    context_window=self.max_chunk_length,
                ),
                ByteLevelTransformerBlock(
                    input_dim=self.byte_hidden,
                    output_dim=self.byte_hidden*4,
                    ffn_dim=self.byte_hidden*8,
                    context_window=self.max_chunk_length,
                ),
                ByteLevelTransformerBlock(
                    input_dim=self.byte_hidden*4,
                    output_dim=self.hidden_dim,
                    ffn_dim=self.byte_hidden*12,
                    context_window=self.max_chunk_length,
                ),
            ]
        )



    def forward(self, x, attention_masks, attention_spans, output_token_ids, end_token_id, pad_token_id):
        """
        Args:
            x: Tensor of shape [batch_size, sequence_length, 384]
            attention_masks: Tensor for attention masks (shape depends on transformer)
            attention_spans: List of length batch_size, each element is a tuple of two tensors:
                            (start_indices, end_indices), each of shape [global_sequence_length_i]
            output_token_ids: Tensor of shape [batch_size, sequence_length, 1]
            end_token_id: Integer representing the end-of-sequence token ID
            pad_token_id: Integer representing the padding token ID
        Returns:
            span_avgs: Tensor of shape [batch_size, max_num_spans, 384] containing average embeddings
            target_tensor: Tensor of shape [batch_size, max_num_spans, max_span_length + 1] containing target token IDs
            target_mask: Tensor of shape [batch_size, max_num_spans, max_span_length + 1] indicating valid tokens
        """
        for block in self.transformer:
            x = block(
                x=x,
                attention_mask=attention_masks
            )

        # x: [batch_size, sequence_length, 384]
        batch_size, sequence_length, embedding_dim = x.size()

        # -------------------------------
        # Handling attention_spans
        # -------------------------------

        # Determine the maximum number of spans across all batches
        # Each span is a tuple (start_indices, end_indices)
        max_num_spans = max(span[0].size(0) for span in attention_spans)

        # Initialize tensors for start and end indices with padding
        # Initialize with 0 indices; masked later
        start_indices_padded = torch.zeros(batch_size, max_num_spans, dtype=torch.long, device=x.device)
        end_indices_padded = torch.zeros(batch_size, max_num_spans, dtype=torch.long, device=x.device)
        span_mask = torch.zeros(batch_size, max_num_spans, dtype=torch.bool, device=x.device)

        for i, span in enumerate(attention_spans):
            start_tensor, end_tensor = span  # Unpack the tuple
            num_spans = start_tensor.size(0)  # Number of spans for this batch

            # Assign start and end indices
            start_indices_padded[i, :num_spans] = start_tensor
            end_indices_padded[i, :num_spans] = end_tensor

            # Mark valid spans in the mask
            span_mask[i, :num_spans] = 1  # Valid spans

        # Compute cumulative sum along the sequence dimension for efficient span sum calculation
        # x: [batch_size, sequence_length, 384]
        cumsum_x = torch.cumsum(x, dim=1)  # [batch_size, sequence_length, 384]

        # Prepend a zero vector at the beginning to handle spans starting at index 0
        zero_padding = torch.zeros(batch_size, 1, embedding_dim, device=x.device)
        cumsum_padded = torch.cat([zero_padding, cumsum_x], dim=1)  # [batch_size, sequence_length + 1, 384]

        # Clamp end and start indices to be within [0, sequence_length]
        end_indices_clamped = end_indices_padded.clamp(max=sequence_length)
        start_indices_clamped = start_indices_padded.clamp(max=sequence_length)

        # Expand indices to match embedding dimensions for gather
        end_indices_expanded = end_indices_clamped.unsqueeze(-1).expand(-1, -1, embedding_dim)
        start_indices_expanded = start_indices_clamped.unsqueeze(-1).expand(-1, -1, embedding_dim)

        # Gather the cumsum vectors
        sum_end = torch.gather(cumsum_padded, dim=1, index=end_indices_expanded)  # [batch_size, max_num_spans, 384]
        sum_start = torch.gather(cumsum_padded, dim=1, index=start_indices_expanded)  # [batch_size, max_num_spans, 384]

        # Compute the sum for each span
        span_sums = sum_end - sum_start  # [batch_size, max_num_spans, 384]

        # Compute span lengths, ensuring no division by zero
        span_lengths = (end_indices_clamped - start_indices_clamped).clamp(min=1).unsqueeze(-1)  # [batch_size, max_num_spans, 1]
        span_lengths = span_lengths.type_as(x)  # Ensure same dtype for division

        # Compute average embeddings
        span_avgs = span_sums / span_lengths  # [batch_size, max_num_spans, 384]

        # Zero out the padded spans
        span_avgs = span_avgs * span_mask.unsqueeze(-1).type_as(x)  # [batch_size, max_num_spans, 384]

        # -------------------------------
        # Processing output_token_ids
        # -------------------------------

        # output_token_ids: [batch_size, sequence_length, 1]
        # Determine the maximum span length across all spans in the batch
        max_span_length = 0
        for span in attention_spans:
            start_tensor, end_tensor = span
            lengths = end_tensor - start_tensor  # Tensor of shape [num_spans]
            if lengths.numel() > 0:
                max_span_length_batch = lengths.max().item()
                if max_span_length_batch > max_span_length:
                    max_span_length = max_span_length_batch

        # Add 1 for the end-of-sequence token
        max_span_length += 1  # Total target length per span

        # Initialize tensors for padded target token IDs and their masks
        # Shape: [batch_size, max_num_spans, max_span_length]
        target_tensor = torch.full(
            (batch_size, max_num_spans, max_span_length),
            pad_token_id,
            dtype=output_token_ids.dtype,
            device=x.device
        )

        # Initialize target mask
        target_mask = torch.zeros(batch_size, max_num_spans, max_span_length, dtype=torch.bool, device=x.device)

        for i, span in enumerate(attention_spans):
            start_tensor, end_tensor = span  # Unpack the tuple
            num_spans = start_tensor.size(0)
            for j in range(num_spans):
                start = start_tensor[j].item()
                end = end_tensor[j].item()
                span_length = end - start  # Actual span length

                if span_length > 0:
                    # Extract token_ids for the span
                    #tokens = output_token_ids[i, start:end]  # [span_length]
                    
                    target_tensor[i, j, :span_length] = output_token_ids[i, start:end]
                    target_tensor[i, j, span_length] = end_token_id
                    # input(target_tensor[i, j])
                    # # Determine the length to copy, considering max_span_length - 1 (for EOS)
                    # copy_length = min(span_length, max_span_length - 1)  # Reserve last position for EOS

                    # if copy_length > 0:
                    #     # Assign tokens to target_tensor
                    #     target_tensor[i, j, :copy_length] = tokens[:copy_length]
                    #     # Update target_mask for valid tokens
                    #     target_mask[i, j, :copy_length] = 1

                    # # Append end_token_id immediately after the actual tokens
                    # eos_position = copy_length
                    # if eos_position < max_span_length:
                    #     target_tensor[i, j, eos_position] = end_token_id
                    #     target_mask[i, j, eos_position] = 1

                else:
                    # If end <= start, only set the EOS token at position 0
                    target_tensor[i, j, 0] = end_token_id
                    target_mask[i, j, 0] = 1

        # Zero out the padded spans in target_tensor and target_mask
        # For spans that are padded (span_mask == 0), set target_tensor to pad_token_id and target_mask to 0
        target_tensor = target_tensor * span_mask.unsqueeze(-1).type_as(x) + \
                        (1 - span_mask.unsqueeze(-1).type_as(x)) * pad_token_id
        target_mask = target_mask * span_mask.unsqueeze(-1)

        return span_avgs, target_tensor, target_mask



class ByteLevelEmbedder(EmbedderInterface):
    """
    Input is a sequence of byte-level token ids
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        self.max_chunk_length = self.model_cfg["max_chunk_length"]
        self.max_num_chunks = self.model_cfg["max_num_chunks"]
        self.byte_hidden = self.model_cfg["byte_hidden"]
        self.hidden_dim = self.model_cfg["hidden_dim"]
        self.num_delimiter_layers = self.model_cfg["num_delimiter_layers"]


        self.byte_tokenizer = build_tokenizer(
            tokenizer_type="bpe",
            vocab_size=model_cfg["vocab_size"],
            simplify=False,
            dataset_name="simple_en_wiki",
        )

        self.byte_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["byte_hidden"],
            device="cuda"
        )

        self.delimiter_model = TokenizerEncoder(
            byte_hidden=self.byte_hidden,
            num_delimiter_layers=self.num_delimiter_layers,
        )

        self.word_encoding_model = ByteBidirectionEncoding(
            max_chunk_length=self.max_chunk_length,
            byte_hidden=self.byte_hidden,
            hidden_dim=self.hidden_dim, 
        )




        # Store pad_token_id and eot_token as class attributes
        self.pad_token_id = self.byte_tokenizer.pad_token
        self.eot_token = self.byte_tokenizer.eot_token

    def forward(self, x):
        # x: (batch_size, seq_len)
        x_embedded = self.byte_embedder(x)  # (batch_size, seq_len, byte_hidden)

        # Pass through delimiter model
        x_transformed, output_token_ids, avg_chunk_len, attention_masks, chunk_spans, chunk_len_loss = self.delimiter_model(
            x=x_embedded,
            x_ids=x,
        )

        # input(attention_masks.size()) # (1, 4096, 4096)

        # Pass through word encoding model
        #x_encoded = self.word_encoding_model(x_transformed, attention_masks, chunk_spans, output_token_ids)  # (batch_size, max_num_chunks, hidden_dim)
        x_encoded, target_tensor, target_mask = self.word_encoding_model(
            x=x_transformed, 
            attention_masks=attention_masks, 
            attention_spans=chunk_spans, 
            output_token_ids=output_token_ids, 
            end_token_id=self.byte_tokenizer.eot_token, 
            pad_token_id=self.byte_tokenizer.pad_token
        )
        # Optionally, pad x_encoded if needed (already handled in ByteBidirectionEncoding)
        # input(f"span_avgs: {span_avgs.size()}")
        # input(f"target_tensor: {target_tensor.size()}")
        # input(f"target_mask: {target_mask.size()}")
        # print(target_tensor)
        # exit()
        return x_encoded, target_tensor, avg_chunk_len, target_mask, chunk_len_loss

    def tokenize_input(self, input_string, truncate=False, add_eot=True):
        token_ids = self.byte_tokenizer.encode(input_string)
        if add_eot:
            token_ids.append(self.eot_token)
        return token_ids

    def decode(self, tokens):
        """ Decode a tensor of tokens into a string. """
        return self.byte_tokenizer.decode_batch(tokens)

