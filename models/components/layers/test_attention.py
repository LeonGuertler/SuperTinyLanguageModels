"""
Pytest functions for the individual components of the different tokenizers.
"""

import pytest
import torch
from models.components.layers import attention

# test the GPT2tokenizer
test_string = "This is a test string. <|endoftext|>"


def test_normal_attention():
    """Just test that it runs..."""
    attention_head = attention.CausalSelfAttention(
        hidden_dim=64,
        num_heads=8,
        bias=False,
        use_rope=True,
        max_context_window=512,
        group_size=1,
    ).to("cuda")
    x = torch.rand(2, 10, 64).to("cuda")
    out = attention_head(x)

    assert out.shape == (2, 10, 64)


def test_group_attention():
    """Just test that it runs with diff groups"""
    attention_head = attention.CausalSelfAttention(
        hidden_dim=64,
        num_heads=8,
        bias=False,
        use_rope=True,
        max_context_window=512,
        group_size=2,
    ).to("cuda")
    x = torch.rand(2, 10, 64).to("cuda")
    out = attention_head(x)

    assert out.shape == (2, 10, 64)
