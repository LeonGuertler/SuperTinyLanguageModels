"""
Pytest functions for the individual components of the different attention.
"""
import pytest 
import torch 

from models.components.layers.attention import (
    build_attention
)

def test_normal_causal_attention():
    """ 
    Just test that it runs and 
    return the correct shape
    """
    attention = build_attention(
        hidden_dim=64,
        context_window=16,
        attn_cfg={
            "attn_type": "generic",
            "num_heads": 8,
            "bias": False,
            "use_rope": False,
            "is_causal": True,
            "group_size": 1,
        }
    )
    x = torch.randn(2, 16, 64)
    y = attention(x)

    assert y.shape == (2, 16, 64)

def test_rope_causal_attention():
    """ 
    Just test that it runs and 
    return the correct shape
    """
    attention = build_attention(
        hidden_dim=64,
        context_window=16,
        attn_cfg={
            "attn_type": "generic",
            "num_heads": 8,
            "bias": False,
            "use_rope": True,
            "is_causal": True,
            "group_size": 1,
        }
    )
    x = torch.randn(2, 16, 64)
    y = attention(x)

    assert y.shape == (2, 16, 64)

def test_normal_bidirectional_attention():
    """ 
    Just test that it runs and 
    return the correct shape
    """
    attention = build_attention(
        hidden_dim=64,
        context_window=16,
        attn_cfg={
            "attn_type": "generic",
            "num_heads": 8,
            "bias": False,
            "use_rope": True,
            "is_causal": False,
            "group_size": 1,
        }
    )
    x = torch.randn(2, 16, 64)
    y = attention(x)

    assert y.shape == (2, 16, 64)


def test_grouped_bidirectional_attention():
    """ 
    Just test that it runs and 
    return the correct shape
    """
    attention = build_attention(
        hidden_dim=64,
        context_window=16,
        attn_cfg={
            "attn_type": "generic",
            "num_heads": 8,
            "bias": False,
            "use_rope": False,
            "is_causal": False,
            "group_size": 2,
        }
    )
    x = torch.randn(2, 16, 64)
    y = attention(x)

    assert y.shape == (2, 16, 64)


def test_grouped_roped_attention():
    """ 
    Just test that it runs and 
    return the correct shape
    """
    attention = build_attention(
        hidden_dim=64,
        context_window=16,
        attn_cfg={
            "attn_type": "generic",
            "num_heads": 8,
            "bias": False,
            "use_rope": True,
            "is_causal": True,
            "group_size": 2,
        }
    )
    x = torch.randn(2, 16, 64)
    y = attention(x)

    assert y.shape == (2, 16, 64)