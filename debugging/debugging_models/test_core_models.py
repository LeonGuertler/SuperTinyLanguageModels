"""
Pytest for core models.
"""

import pytest
import torch

from models.build_model import build_core_model


def test_generic_core_model():
    """
    Test the generic core model.
    hidden_dim: 512
    context_window: 512
    vocab_size: 50257
    core_model: baseline
    embedder: baseline
    lm_head: baseline
    """
    model = build_core_model(
        model_cfg={
            "hidden_dim": 64,
            "context_window": 64,
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
                    "use_rope": False,
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
