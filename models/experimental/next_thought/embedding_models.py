"""
The Embedding model for a VAE style sequence to sequence model.
"""
import torch 
import numpy as np

from models.embedding_models import GenericEmbedder
from models.components.layers.transformer_blocks import GenericTransformerBlock

from models.components.positional_encoding import build_positional_encodings
from models.components.tokenizers import build_tokenizer


# import local components
from models.experimental.next_thought.layers import AttentionPoolingRemoval



class HierarchicalEncoder(GenericEmbedder):
    """
    Accepts an arbitrary length sequence as input,
    uses the QK^T matrix to, at every layer,
    pick the top n-percent of nodes to pool into 
    a single token (the one paying most attention 
    to the other should be pooled into the other token).
    """
    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)
        # build the tokenizer
        self.tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["embedder"]["tokenizer_type"],
            vocab_size=model_cfg["vocab_size"],
            dataset_name=model_cfg["embedder"]["dataset_name"],
        )

        # build the token embeddings
        self.token_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["embedder"]["pooling_dims"][0],
        )

        # build the positional encodings
        self.positional_encodings = build_positional_encodings(model_cfg=model_cfg)


        self.standard_transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=model_cfg["embedder"]["pooling_dims"][0],
                    context_window=model_cfg["embedder"]["context_window"],
                    use_rope=False,
                    ffn_cfg=model_cfg["embedder"]["standard_ffn_block"],
                    attn_cfg=model_cfg["embedder"]["standard_attn_block"],
                )
            ]
        )

        self.pooling_transformer = torch.nn.ModuleList(
            [

                AttentionPoolingRemoval(
                    hidden_size_in=model_cfg["embedder"]["pooling_dims"][i],
                    hidden_size_out=model_cfg["embedder"]["pooling_dims"][i+1],
                    num_attention_heads=12,
                    pct_pool_per_layer=model_cfg["embedder"]["pooling_pct_per_layer"][i],
                ) for i in range(len(model_cfg["embedder"]["pooling_dims"]) - 1)
            ]
        )

        self.context_window = model_cfg["embedder"]["context_window"]


    def forward(self, token_ids):
        # embed the input 
        input(token_ids)
        x = self.token_embedder(token_ids)

        # apply positional encoding 
        x = x + self.positional_encodings(x)


        # first pass through normal attention blocks
        for layer in self.standard_transformer:
            x = layer(x)

        # then pass through pooling attention blocks
        for layer in self.pooling_transformer:
            x = layer(x)
        # mean pool final representation
        x = x.mean(dim=-2)
        return x
    

    def tokenize_input(self, input_string):
        """
        Tokenize the input
        """
        # truncate
        token_ids = np.ones(self.context_window, dtype=np.uint16) * self.tokenizer.pad_token
        raw_token_ids = self.tokenizer.encode(input_string)
        end_id = min(len(raw_token_ids), self.context_window) - 1
        token_ids[:end_id] = raw_token_ids[:end_id]
        token_ids[end_id] = self.tokenizer.eot_token
        return token_ids