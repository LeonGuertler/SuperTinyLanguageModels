"""
Pytest functions for the individual components of the different normalization layers.
"""

# pylint: disable=unused-import
import pytest

# pylint: enable=unused-import
import torch

from models.components.layers.normalization import build_normalization


def test_rms_norm():
    """Just test that it runs..."""
    normalization = build_normalization(
        normalization_name="rms_norm",
        dim=64,
        bias=False
    )
    x = torch.rand(2, 10, 64)
    out = normalization(x)

    assert out.shape == (2, 10, 64)


def test_layer_norm():
    """Just test that it runs..."""
    normalization = build_normalization(
        normalization_name="layer_norm",
        dim=64,
        bias=False
    )
    x = torch.rand(2, 10, 64)
    out = normalization(x)

    assert out.shape == (2, 10, 64)