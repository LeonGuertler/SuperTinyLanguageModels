"""
Pytest for the full transformer blocks.
"""

import pytest
import torch

from models.components.layers.transformer_blocks import GenericTransformerBlock


def test_generic_transformer_block():
    """
    Simple test to check that the generic transformer block
    """
    block = GenericTransformerBlock(
        hidden_dim=64,
        context_window=16,
        ffn_cfg={
            "ffn_type": "generic",
            "ffn_dim": 128,
            "bias": False,
            "activation": "relu",
            "normalization": "layer_norm",
        },
        attn_cfg={
            "attn_type": "generic",
            "num_heads": 8,
            "bias": False,
            "use_rope": False,
            "is_causal": True,
            "group_size": 1,
            "normalization": "rms_norm",
        },
        norm_bias=False,
    )
    x = torch.randn(2, 16, 64)
    y = block(x)

    assert y.shape == (2, 16, 64)
