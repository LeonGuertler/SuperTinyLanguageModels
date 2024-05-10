"""
Tests the different feedforward layers.
"""
import pytest 
import torch 
from models.components.layers.feedforward import (
    build_ffn
)

def test_generic():
    """
    Simple test to check that the generic feedforward
    """
    ffn = build_ffn(
        hidden_dim=64,
        ffn_cfg={
            "ffn_type": "generic",
            "ffn_dim": 128,
            "bias": False,
            "activation": "relu"
        }
    )
    x = torch.randn(2, 16, 64)
    y = ffn(x)

    assert y.shape == (2, 16, 64)

def test_swiglue():
    """
    Simple test to check that the swiglue feedforward
    """
    ffn = build_ffn(
        hidden_dim=64,
        ffn_cfg={
            "ffn_type": "swiglue",
            "ffn_dim": 128,
            "bias": False,
        }
    )
    x = torch.randn(2, 16, 64)
    y = ffn(x)

    assert y.shape == (2, 16, 64)

