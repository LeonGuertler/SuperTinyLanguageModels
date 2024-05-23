"""
The latent to variable length sequence decoder.
"""
import torch 

from models.experimental.next_thought.layers import (
    LatentSpaceDecoder, 
    LatentSpaceQuery
)

from models.embedding_models import GenericEmbedder
from models.components.layers.transformer_blocks import GenericTransformerBlock
from models.components.positional_encoding import build_positional_encodings



class VariableLengthLatentDecoder(torch.nn.Module):
    """
    Given a latent space representation, decode it into a sequence.
    This should be similar to how VLMs work (i.e. have an encoder
    for the latent space and query it at each step to generate the
    next token).
    """
    def __init__(self, model_cfg, embedding_model):
        super().__init__()
        self.model_cfg = model_cfg
        self.latent_decoder = torch.nn.Linear(
            in_features=model_cfg["latent_dim"],
            out_features=model_cfg["embedding_dim"] * model_cfg["lm_head"]["latent_decoded_into"],
            bias=False
        )

        self.token_embedder = embedding_model.token_embedder
        self.positional_encodings = embedding_model.positional_encodings

        self.autoregressive_transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=model_cfg["embedding_dim"],
                    context_window=model_cfg["context_window"],
                    use_rope=False,
                    ffn_cfg=model_cfg["lm_head"]["standard_ffn_block"],
                    attn_cfg=model_cfg["lm_head"]["standard_attn_block"],
                ) for _ in range(model_cfg["lm_head"]["num_layers"])
            ]
        )

        self.lm_head = torch.nn.Linear(
            in_features=model_cfg["embedding_dim"],
            out_features=model_cfg["vocab_size"],
            bias=False
        )

    
    def forward(self, x, y=None):
        """
        forward
        """
        # decode latent into tokens
        x = self.latent_decoder(x)
        # reshape
        x = x.view(x.size(0), self.model_cfg["lm_head"]["latent_decoded_into"], self.model_cfg["embedding_dim"])

        # encode the target tokens with the embedder (w/o gradient)
        y = self.token_embedder(y)

        # add positional encoding
        y = y + self.positional_encodings(y)

        # concat with the latent tokens
        x = torch.cat([x, y], dim=1)

        # pass through autoregressive transformer blocks
        for layer in self.autoregressive_transformer:
            x = layer(x)

        # pass through lm head
        x = self.lm_head(x[:, self.model_cfg["lm_head"]["latent_decoded_into"]:])

        return x

