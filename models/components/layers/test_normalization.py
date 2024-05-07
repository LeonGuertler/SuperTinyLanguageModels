"""
Pytest functions for the individual components of the different normalization layers.
"""

# pylint: disable=unused-import
import pytest

# pylint: enable=unused-import
import torch

from models.components.layers import normalization


def test_residual_layer_norm():
    """Just test that it runs..."""
    x = torch.rand(2, 10, 64)
    out = normalization.RMSNorm(dim=64)(x)

    assert out.shape == (2, 10, 64)
