"""
Pytest functions for the individual components of the different tokenizers.
"""

# pylint: disable=unused-import
import pytest

# pylint: enable=unused-import
import torch

from models.components.layers import attention

# test the GPT2tokenizer
TEST_STRING = "This is a test string. <|endoftext|>"


def test_normal_attention():
    """Just test that it runs..."""
    attention_head = attention.SelfAttention(
        hidden_dim=64,
        num_heads=8,
        bias=False,
        use_rope=True,
        max_context_window=512,
    )
    x = torch.rand(2, 10, 64)
    out = attention_head(x)

    assert out.shape == (2, 10, 64)


def test_group_attention():
    """Just test that it runs with diff groups"""
    attention_head = attention.SelfAttention(
        hidden_dim=64,
        num_heads=8,
        bias=False,
        use_rope=True,
        max_context_window=512,
    )
    x = torch.rand(2, 10, 64)
    out = attention_head(x)

    assert out.shape == (2, 10, 64)
