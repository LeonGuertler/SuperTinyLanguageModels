"""
Pytest for core models.
"""
import pytest 
import torch 

from models.build_model import (
    build_core_model
)

def test_generic_core_model():
    """
    Test the generic core model.
    """
    model = build_core_model(
        model_cfg={
            "core_model_type": "generic",
            "hidden_dim": 64,
            "context_window": 16,
            "bias": True,
            "ffn": {
                "ffn_type": "generic",
                "ffn_dim": 128,
                "activation": "relu",
                "normalization": "layer_norm",
                "bias": True
            },
            "attn": {
                "attn_type": "generic",
                "num_heads": 8,
                "normalization": "layer_norm",
                "group_size": 8,
                "bias": True,
                "use_rope": False,
                "is_causal": False,
            },
            "num_layers": 2,
        }
    )

    x = torch.randn(2, 16, 64)
    y = model(x)

    assert y.shape == (2, 16, 64)