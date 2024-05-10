"""
Test the positional encoding
"""

import pytest
import torch

from models.components.positional_encoding import build_positional_encodings


def test_learned_positional_encodings():
    """
    Test that the learned positional encodings
    return the correct shape.
    """
    pos_encodings = build_positional_encodings(
        model_cfg={
            "positional_encoding_type": "learned",
            "hidden_dim": 64,
            "context_window": 16,
        }
    )

    x = torch.randn(2, 16, 64)
    y = pos_encodings(x)

    assert y.shape == (2, 16, 64)
