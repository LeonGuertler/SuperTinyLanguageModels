"""
A collection of embedding models. A collection model includes
the tokenizer(s), token embeddings and positional encodings
(if necessary).
"""

import torch

from models.components.positional_encoding import LearnedPosEncoding
from models.embedding_models import GenericEmbedder
from models.components.tokenizers import build_tokenizer
from models.experimental.byte_level.layers import ByteLevelTransformerBlock


class ByteLevelEmbedder(GenericEmbedder):
    """
    Takes byte level encodings, processes them via
    two local-attention transformer blocks and pools
    the resultant tokens based on gpt-2 tokenizer
    boundaries.
    Inputs are batches of lists of token blocks
    in the gpt2 tokenizer boundaries.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)
        self.model_cfg = model_cfg

        # build the tokenizers
        self.byte_tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["embedder"]["byte_tokenizer_type"],
            vocab_size=model_cfg["byte_vocab_size"],
            dataset_name=model_cfg["embedder"]["dataset_name"],
        )
        self.pooling_tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["embedder"]["tokenizer_type"],
            vocab_size=model_cfg["vocab_size"],
            dataset_name=model_cfg["embedder"]["dataset_name"],
        )

        # positional encodings
        self.pos_encoder = LearnedPosEncoding(
            hidden_dim=model_cfg["byte_embedding_dim"],
            context_window=model_cfg["byte_context_window"],
        )

        # build the token embeddings
        self.byte_token_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["byte_vocab_size"],
            embedding_dim=model_cfg["byte_embedding_dim"],
        )

        # build the transformer blocks
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_embedding_dim"],
                    output_dim=model_cfg["byte_embedding_dim"] * 2,
                    ffn_dim=model_cfg["byte_embedding_dim"] * 4,
                    context_window=model_cfg["byte_context_window"],
                    use_rope=False,
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_embedding_dim"] * 2,
                    output_dim=model_cfg["hidden_dim"],
                    ffn_dim=model_cfg["byte_embedding_dim"] * 8,
                    context_window=model_cfg["byte_context_window"],
                    use_rope=False,
                ),
            ]
        )

    # pylint: enable=super-init-not-called

    def tokenize_input(self, input_string):
        """Tokenize an input string.

        In this case we actually want to pre-tokenize using the pooling tokenizer,
        the byte tokenizer is then used in the forward pass. Its a bit complicated...
        """
        pooling_ids = self.pooling_tokenizer.encode(input_string)
        tokens = [
            self.byte_tokenizer.encode(self.pooling_tokenizer.decode([pool_id]))
            for pool_id in pooling_ids
        ]
        # truncate
        tokens = [
            token_seq[: self.model_cfg["byte_context_window"]] for token_seq in tokens
        ]
        # pad
        tokens = [
            token_seq
            + [self.byte_tokenizer.pad_token]
            * (self.model_cfg["byte_context_window"] - len(token_seq))
            for token_seq in tokens
        ]
        return tokens

    def forward(self, token_ids):
        """
        Forward pass.
        """
        print(token_ids.size())
        # get the byte embeddings
        x = self.byte_token_embedder(token_ids)
        input(x.size())

        # positional encoding
        x = self.pos_encoder(x)

        # pass through transformer
        for block in self.transformer:
            x = block(x)

        return x
