"""
Init to simplify imports.
"""

from torch import nn

from models.components.layers.attention import SelfAttention
from models.components.layers.blocks import (
    BaseTransformerBlock,
    BidirectionalTransformerBlock,
    JetFFNMoEBlock,
    ModernTransformerBlock,
)
from models.components.layers.feedforward import FFN, SWIGluFFN
from models.components.layers.moe import MoE
from models.components.layers.normalization import LayerNorm, build_normalization
