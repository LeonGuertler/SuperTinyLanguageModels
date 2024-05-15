"""
Pytest for core models.
"""

import pytest
import torch

from models.build_models import build_core_model


def test_generic_core_model():
    """
    Test the generic core model.
    """
    model = build_core_model(
        model_cfg={
            "hidden_dim": 64,
            "context_window": 64,
            "vocab_size": 50257,
            "positional_embedding_type": "rope",
            "core_model": {
                "core_model_type": "generic",
                "norm_bias": True,
                "ffn": {
                    "ffn_type": "generic",
                    "ffn_dim": 128,
                    "activation": "relu",
                    "normalization": "layer_norm",
                    "bias": True,
                },
                "attn": {
                    "attn_type": "generic",
                    "num_heads": 8,
                    "group_size": 8,
                    "bias": True,
                    "is_causal": False,
                    "normalization": "rms_norm",
                },
                "num_layers": 2,
            },
        }
    )

    x = torch.randn(2, 16, 64)
    y = model(x)

    assert y.shape == (2, 16, 64)
