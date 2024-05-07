"""Tests for the feedforward layers."""

# pylint: disable=unused-import
import pytest

# pylint: enable=unused-import
import torch

from models.components.layers import feedforward


def test_ffn():
    """Just test that it runs..."""
    ffn = feedforward.FFN(hidden_dim=64, ffn_dim=128)
    x = torch.rand(2, 10, 64)
    out = ffn(x)

    assert out.shape == (2, 10, 64)


def test_swiglu():
    """Just test that it runs..."""
    ffn = feedforward.SWIGluFFN(hidden_dim=64, ffn_dim=128)
    x = torch.rand(2, 10, 64)
    out = ffn(x)

    assert out.shape == (2, 10, 64)


def test_jet_moe_ffn():
    """Just test that it runs..."""
    ffn = feedforward.JetMoEFFN(
        hidden_dim=64, ffn_dim=128, num_experts=4, top_k=2, bias=False
    ).eval()
    x = torch.rand(2, 10, 64)
    out, _ = ffn(x)

    assert out.shape == (2, 10, 64)
