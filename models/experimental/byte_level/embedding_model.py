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

    def __init__(self, max_chunk_length):
        super().__init__()

        layers = 3
        self.max_chunk_length = max_chunk_length

        self.transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=64,
                    context_window=5 * 2048,
                    use_rope=True,
                    ffn_cfg={
                        "ffn_type": "generic",
                        "ffn_dim": 4 * 64,
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
                for _ in range(layers)
            ]
        )

        self.end_of_seq_head = torch.nn.Linear(
            64,
            2,  # 2 because we just need to predict whether it's a <sep> or <end>
            bias=True,
        )

    def forward(self, x, pad_token_vector, x_ids, pad_token_id):
        # Pass through transformer blocks
        x_transformed = x
        for block in self.transformer:
            x_transformed = block(x_transformed)

        # Pass through end_of_seq_head
        output = self.end_of_seq_head(x_transformed)  # Shape: (batch, seq_len, 2)

        # Determine where to split chunks based on output logits
        end_of_chunk = output[..., 0] > output[..., 1]  # Shape: (batch, seq_len)

        batch_size, seq_len = end_of_chunk.size()
        device = x.device

        # Find chunk end indices for each sequence in the batch
        chunk_indices = []
        avg_chunk_len = []
        for batch in range(batch_size):
            ends = torch.nonzero(end_of_chunk[batch], as_tuple=False).squeeze(-1)
            if ends.numel() == 0:
                ends = torch.tensor([seq_len - 1], device=device)
            starts = torch.cat([torch.tensor([0], device=device), ends[:-1] + 1])
            chunk_indices.append((starts, ends))
            avg_chunk_len.append((ends - starts).float().mean())

        print(f"Average Chunk len: {sum(avg_chunk_len)/len(avg_chunk_len)}")
        reg_term = output[:, :, 1].mean()
        max_num_chunks = 1024
        max_chunk_length = self.max_chunk_length

        # Initialize output tensors by repeating pad_token_vector
        # pad_token_vector shape: (1, 128)
        # Desired shape: (batch_size, max_num_chunks, max_chunk_length, 128)
        output_tensor = pad_token_vector.repeat(batch_size, max_num_chunks, max_chunk_length, 1)

        # Initialize output_token_ids with pad_token_id
        output_token_ids = torch.full(
            (batch_size, max_num_chunks, max_chunk_length),
            pad_token_id,
            device=device,
            dtype=torch.long,
        )

        # Populate output_tensor and output_token_ids with actual chunk data
        for batch in range(batch_size):
            starts, ends = chunk_indices[batch]
            num_chunks = min(len(ends), max_num_chunks)
            for i in range(num_chunks):
                start = starts[i]
                end = ends[i] + 1  # Include the end index
                chunk = x_transformed[batch, start:end, :]  # Shape: (chunk_len, 128)
                chunk_ids = x_ids[batch, start:end]

                chunk_len = chunk.size(0)
                if chunk_len > max_chunk_length:
                    chunk = chunk[:max_chunk_length]
                    chunk_ids = chunk_ids[:max_chunk_length]
                    chunk_len = max_chunk_length

                output_tensor[batch, i, :chunk_len, :] = chunk
                output_token_ids[batch, i, :chunk_len] = chunk_ids

        return output_tensor, output_token_ids, reg_term #sum(avg_chunk_len)/len(avg_chunk_len)




class ByteBidirectionEncoding(torch.nn.Module):
    """
    Input shape: batch x max_num_chuncks x max_chunck_length x 128 (byte_embed_dim)
    return batch x max_num_chuncks x hidden_dim (512)
    """
    def __init__(self, max_chunk_length):
        super().__init__()
        # build the transformer blocks
        hidden = 64
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=hidden,
                    output_dim=hidden,
                    ffn_dim=hidden*4,
                    context_window=max_chunk_length,
                    use_rope=True,
                ),
                ByteLevelTransformerBlock(
                    input_dim=hidden,
                    output_dim=hidden ,
                    ffn_dim=hidden*4,
                    context_window=max_chunk_length,
                    use_rope=True,
                ),
                ByteLevelTransformerBlock(
                    input_dim=hidden,
                    output_dim=hidden*4,
                    ffn_dim=hidden*8,
                    context_window=max_chunk_length,
                    use_rope=True,
                ),
                ByteLevelTransformerBlock(
                    input_dim=hidden*4,
                    output_dim=hidden*8,
                    ffn_dim=hidden*12,
                    context_window=max_chunk_length,
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

        # use last # TODO try mean pooling
        #x = x[:, -1, :] # (batch*C_num), 1, 512 
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
        max_chunk_length = 12

        self.byte_tokenizer = build_tokenizer(
            tokenizer_type="bpe",
            vocab_size=model_cfg["vocab_size"],
            simplify=False,
            dataset_name="simple_en_wiki",
        )

        self.byte_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=64,
            device="cuda"
        )

        self.delimiter_model = TokenizerEncoder(
            max_chunk_length=max_chunk_length
        )

        self.word_encoding_model = ByteBidirectionEncoding(
            max_chunk_length=max_chunk_length
        )




        # Store pad_token_id and eot_token as class attributes
        self.pad_token_id = self.byte_tokenizer.pad_token
        self.eot_token = self.byte_tokenizer.eot_token

    def forward(self, x):
        # Pass through byte tokenizer (assuming x is token IDs)
        # x = self.byte_tokenizer.encode(token_ids)
        # Register pad_token_vector as a buffer to avoid recomputing
        pad_token_vector = self.byte_embedder(
            torch.tensor([self.byte_tokenizer.pad_token]).to("cuda")
        )
        self.register_buffer("pad_token_vector", pad_token_vector)
        # Pass through delimiter model
        x_embedded = self.byte_embedder(x)
        x, output_token_ids, chunk_len = self.delimiter_model(
            x=x_embedded,
            pad_token_vector=self.pad_token_vector,
            x_ids=x,
            pad_token_id=self.pad_token_id,
        )

        # Pass through word encoding model
        x = self.word_encoding_model(x)

        return x, output_token_ids, chunk_len

    def tokenize_input(self, input_string, truncate=False, add_eot=True):
        token_ids = self.byte_tokenizer.encode(input_string)
        if add_eot:
            token_ids.append(self.eot_token)
        return token_ids
