"""
A collection of embedding models. A collection model includes
the tokenizer(s), token embeddings and positional encodings
(if necessary).
"""

import torch

from models.components.positional_encoding import LearnedPosEncoding
from models.components.layers.tokenizers import build_tokenizer
from models.embedding_models import EmbedderInterface
from models.experimental.byte_level.layers import ByteLevelTransformerBlock

from models.components.layers.transformer_blocks import GenericTransformerBlock
from copy import deepcopy

class TokenizerEncoder(torch.nn.Module):
    """
    Take seq of byte embeddings, return sequence of delimiters. (binary)
    """

    def __init__(self, num_delimiter_layers, byte_hidden, max_chunk_length, max_num_chunks):
        super().__init__()

        self.num_delimiter_layers = num_delimiter_layers
        self.max_chunk_length = max_chunk_length
        self.max_num_chunks = max_num_chunks
        self.byte_hidden = byte_hidden



        self.transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=self.byte_hidden,
                    context_window=5 * 2048,
                    use_rope=True,
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
                    },
                )
                for _ in range(self.num_delimiter_layers)
            ]
        )

        self.end_of_seq_head = torch.nn.Linear(
            self.byte_hidden,
            1,  # 2 because we just need to predict whether it's a <sep> or <end>
            bias=True,
        )

    def forward(self, x, pad_token_vector, x_ids, pad_token_id):
        # Pass through transformer blocks
        x_transformed = x
        for block in self.transformer:
            x_transformed = block(x_transformed)

        # Pass through end_of_seq_head
        # output = self.end_of_seq_head(x_transformed)  # Shape: (batch, seq_len, 2)
        logits = self.end_of_seq_head(x_transformed).squeeze(-1)  # Shape: (batch, seq_len)

        # Apply sigmoid activation
        probs = torch.sigmoid(logits)
        # Determine chunk boundaries using a threshold
        threshold = 0.8  # Adjust as needed
        end_of_chunk = probs > threshold

        chunk_loss = torch.mean(torch.pow(probs-0.75, 2))

        batch_size, seq_len = end_of_chunk.size()
        device = x.device

        # Inside TokenizerEncoder.forward()
        chunk_indices = []
        avg_chunk_len = []
        for batch in range(batch_size):
            ends = torch.nonzero(end_of_chunk[batch], as_tuple=False).squeeze(-1)
            if ends.numel() == 0:
                ends = torch.tensor([seq_len - 1], device=device)
            else:
                ends = ends + 1  # Adjust ends to point after the chunk end
            starts = torch.cat([torch.tensor([0], device=device), ends[:-1]])
            chunk_lengths = ends - starts
            # Filter out zero-length chunks
            valid = chunk_lengths > 0
            starts = starts[valid]
            ends = ends[valid]
            chunk_indices.append((starts, ends))
            avg_chunk_len.append(chunk_lengths[valid].float().mean())


        #print(f"Average Chunk len: {sum(avg_chunk_len)/len(avg_chunk_len)}")

        max_chunk_length = self.max_chunk_length

        # Initialize output tensors by repeating pad_token_vector
        # pad_token_vector shape: (1, 128)
        # Desired shape: (batch_size, max_num_chunks, max_chunk_length, 128)
        output_tensor = pad_token_vector.repeat(batch_size, self.max_num_chunks, self.max_chunk_length, 1)

        # Initialize output_token_ids with pad_token_id
        output_token_ids = torch.full(
            (batch_size, self.max_num_chunks, self.max_chunk_length),
            pad_token_id,
            device=device,
            dtype=torch.long,
        )

        # Populate output_tensor and output_token_ids with actual chunk data
        for batch in range(batch_size):
            starts, ends = chunk_indices[batch]
            num_chunks = min(len(ends), self.max_num_chunks)
            for i in range(num_chunks):
                start = starts[i]
                end = ends[i] + 1  # Include the end index
                chunk = x_transformed[batch, start:end, :]  # Shape: (chunk_len, 128)
                chunk_ids = x_ids[batch, start:end]

                chunk_len = chunk.size(0)
                if chunk_len > self.max_chunk_length:
                    chunk = chunk[:self.max_chunk_length]
                    chunk_ids = chunk_ids[:self.max_chunk_length]
                    chunk_len = self.max_chunk_length

                output_tensor[batch, i, :chunk_len, :] = chunk
                output_token_ids[batch, i, :chunk_len] = chunk_ids
        return output_tensor, output_token_ids, chunk_loss, sum(avg_chunk_len)/len(avg_chunk_len)

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
                    context_window=2048, #5 * 2048,
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
        ) # [Hello ][World!] (12, 32) -> (12, 1) 

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

        chunk_len_loss = torch.mean(torch.pow(probs-0.25, 2)) #torch.mean(torch.sum(probs, dim=1)) # change to exp. val. 0.85
        

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

        return x_transformed, x_ids, chunk_len_loss, avg_chunk_len #, attention_masks, chunk_spans, chunk_len_loss # (12, 32), (12, 1), int, (12, 12), List(Tuple(int, int)), 


        # return output_tensor, output_token_ids, chunk_loss, sum(avg_chunk_len)/len(avg_chunk_len)

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
                    use_rope=True,
                ),
                ByteLevelTransformerBlock(
                    input_dim=self.byte_hidden,
                    output_dim=self.byte_hidden ,
                    ffn_dim=self.byte_hidden*4,
                    context_window=self.max_chunk_length,
                    use_rope=True,
                ),
                ByteLevelTransformerBlock(
                    input_dim=self.byte_hidden,
                    output_dim=self.byte_hidden*4,
                    ffn_dim=self.byte_hidden*8,
                    context_window=self.max_chunk_length,
                    use_rope=True,
                ),
                ByteLevelTransformerBlock(
                    input_dim=self.byte_hidden*4,
                    output_dim=self.hidden_dim,
                    ffn_dim=self.byte_hidden*12,
                    context_window=self.max_chunk_length,
                    use_rope=True,
                ),
            ]
        )



    def forward(self, x):
        # B, Num_chunck, Chunck_len, 128 
        B, C_num, C_len, h_b = x.size()
        # flatten first two dims 
        x = x.view(B*C_num, C_len, h_b)

        # pass through blocks
        for block in self.transformer:
            x = block(x)

        x = x.mean(-2)
        # reshape it back to 3
        x = x.view(B, C_num, -1)  # batch, chunk_num, 512

        return x 


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
            max_chunk_length=self.max_chunk_length,
            max_num_chunks=self.max_num_chunks,
            byte_hidden=self.byte_hidden,
            num_delimiter_layers=self.num_delimiter_layers
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
        pad_token_vector = self.byte_embedder(
            torch.tensor([self.byte_tokenizer.pad_token]).to("cuda")
        )
        self.register_buffer("pad_token_vector", pad_token_vector)
        # Pass through delimiter model
        x_embedded = self.byte_embedder(x)
        x, output_token_ids, chunk_loss, avg_chunk_len = self.delimiter_model(
            x=x_embedded,
            pad_token_vector=self.pad_token_vector,
            x_ids=x,
            pad_token_id=self.pad_token_id,
        )

        # Pass through word encoding model
        x = self.word_encoding_model(x)

        return x, output_token_ids, chunk_loss, avg_chunk_len

    def tokenize_input(self, input_string, truncate=False, add_eot=True):
        token_ids = self.byte_tokenizer.encode(input_string)
        if add_eot:
            token_ids.append(self.eot_token)
        return token_ids
