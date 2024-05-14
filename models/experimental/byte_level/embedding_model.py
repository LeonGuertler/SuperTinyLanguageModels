"""
A collection of embedding models. A collection model includes 
the tokenizer(s), token embeddings and positional encodings 
(if necessary).
"""
import torch 

from models.components.tokenizers import build_tokenizer
from models.components.positional_encoding import build_positional_encodings

from models.experimental.byte_level.layers import (
    ProjectingFFN,
    ByteLevelTransformerBlock,
)
from models.components.positional_encoding import LearnedPosEncoding



class ByteLevelEmbedder(torch.nn.Module):
    """
    Takes byte level encodings, processes them via
    two local-attention transformer blocks and pools
    the resultant tokens based on gpt-2 tokenizer
    boundaries.
    Inputs are batches of lists of token blocks
    in the gpt2 tokenizer boundaries.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        # build the tokenizers
        self.byte_tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["byte_tokenizer"],
            vocab_size=model_cfg["byte_vocab_size"],
            dataset_name=model_cfg["tokenizer_dataset"],
        )
        self.pooling_tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["pooling_tokenizer"],
            vocab_size=model_cfg["pooling_vocab_size"],
            dataset_name=model_cfg["tokenizer_dataset"],
        )

        # positional encodings
        self.pos_encoder = LearnedPosEncoding(
            hidden_dim=self.embedding_dim,
            context_window=self.byte_context_window
        )

        # build the token embeddings
        self.byte_token_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["byte_vocab_size"],
            embedding_dim=model_cfg["embedding_dim"],
        )

        # build the transformer blocks
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["embedding_dim"],
                    output_dim=model_cfg["embedding_dim"]*2,
                    ffn_dim=model_cfg["embedding_dim"]*4,
                    context_window=model_cfg["byte_context_window"],
                    use_rope=False,

                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["embedding_dim"]*2,
                    output_dim=model_cfg["hidden_dim"],
                    ffn_dim=model_cfg["embedding_dim"]*8,
                    context_window=model_cfg["byte_context_window"],
                    use_rope=False,
                ),
            ]
        )

    def forward(self, x):
        """
        Forward pass.
        """
        # get the byte embeddings
        x = self.byte_token_embedder(x)

        # positional encoding 
        x = self.pos_encoder(x)

        # pass through transformer
        for block in self.transformer:
            x = block(x)

        return x




