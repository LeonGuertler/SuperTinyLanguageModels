"""
The Embedding model for a VAE style sequence to sequence model.
"""

from typing import Literal

import torch

from models.components.layers import attention, feedforward
from models.components.layers.transformer_blocks import GenericTransformerBlock
from models.components.positional_encoding import build_positional_encodings
from models.components.tokenizers import build_tokenizer
from models.embedding_models import GenericEmbedder, GenericEmbedderConfig

# import local components
from models.experimental.next_thought.layers import AttentionPoolingRemoval


class HierarchicalEncoderConfig(GenericEmbedderConfig):
    """
    Hierarchical Encoder Configuration
    """

    embedder_type: Literal["hierarchical"]
    pooling_dims: list = [768, 256, 128]
    pct_pool_per_layer: list = [0.2, 0.2]
    ffn: feedforward.FFNConfig
    attn: attention.AttentionConfig


class HierarchicalEncoder(GenericEmbedder):
    """
    Accepts an arbitrary length sequence as input,
    uses the QK^T matrix to, at every layer,
    pick the top n-percent of nodes to pool into
    a single token (the one paying most attention
    to the other should be pooled into the other token).
    """

    def __init__(
        self,
        embedder_cfg: HierarchicalEncoderConfig,
        vocab_size: int,
        hidden_dim: int,
        context_window: int,
        positional_encoding_type: str,
    ):
        super().__init__(
            embedder_cfg=embedder_cfg,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            context_window=context_window,
            positional_encoding_type=positional_encoding_type,
        )
        # build the tokenizer
        self.tokenizer = build_tokenizer(
            tokenizer_type=embedder_cfg.tokenizer_type,
            vocab_size=vocab_size,
            dataset_name=embedder_cfg.dataset_name,
        )

        # build the token embeddings
        self.token_embedder = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedder_cfg.pooling_dims[0],
        )

        # build the positional encodings
        self.positional_encodings = build_positional_encodings(
            positional_encoding_type, hidden_dim, context_window
        )

        self.standard_transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=embedder_cfg.pooling_dims[0],
                    context_window=context_window,
                    ffn_cfg=embedder_cfg.ffn,
                    attn_cfg=embedder_cfg.attn,
                )
            ]
        )

        self.pooling_transformer = torch.nn.ModuleList(
            [
                AttentionPoolingRemoval(
                    hidden_size_in=embedder_cfg.pooling_dims[i],
                    hidden_size_out=embedder_cfg.pooling_dims[i + 1],
                    num_attention_heads=12,
                    pct_pool_per_layer=embedder_cfg.pct_pool_per_layer[i],
                )
                for i in range(len(embedder_cfg.pooling_dims) - 1)
            ]
        )

    def forward(self, token_ids):
        # embed the input
        x = self.embedding(token_ids)

        # apply positional encoding
        x = x + self.positional_encoding(x)

        # first pass through normal attention blocks
        for layer in self.standard:
            x = layer(x)

        # then pass through pooling attention blocks
        for layer in self.pooling_attention:
            x = layer(x)
        # mean pool final representation
        x = x.mean(dim=-2)
        return x
