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


