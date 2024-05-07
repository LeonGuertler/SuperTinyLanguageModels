"""Tests for the block layers."""

# pylint: disable=unused-import
import pytest

# pylint: enable=unused-import
import torch

from models.components.layers.blocks import (
    BaseTransformerBlock,
    ModernTransformerBlock,
    MoE,
)


def test_base_transformer_block():
    """Just test that it runs..."""
    block = BaseTransformerBlock(
        hidden_dim=64, ffn_dim=128, ffn_activation="relu", bias=False, num_heads=8
    )
    x = torch.rand(2, 10, 64)
    out = block(x)

    assert out.shape == (2, 10, 64)


def test_modern_transformer_block():
    """Just test that it runs..."""
    block = ModernTransformerBlock(
        hidden_dim=64, ffn_dim=128, num_heads=8, context_window=512
    )
    x = torch.rand(2, 10, 64)
    out = block(x)

    assert out.shape == (2, 10, 64)


def test_moe():
    """Just test that it runs..."""
    block = MoE(
        hidden_dim=64,
        ffn_dim=128,
        activation="relu",
        bias=False,
        num_experts=4,
        top_k=2,
    )
    block.eval()
    x = torch.rand(2, 10, 64)
    out, _ = block(x)

    assert out.shape == (2, 10, 64)
