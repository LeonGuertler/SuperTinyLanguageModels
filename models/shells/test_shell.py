"""Test the model shells."""

# pylint: disable=missing-function-docstring
# pylint: disable=unused-import
import pytest
import torch

from models.core_models import GenericTransformer
from models.shells import (
    AutoregressiveByteModelShell,
    AutoregressiveModelShell,
    build_shell,
)

AUTO_TEST_CONFIG = {
    "model_shell": {
        "shell_type": "autoregressive",
        "tokenizer": "BPE",
        "tokenizer_dataset_name": "debug",
        "embedding_dim": 512,
        "vocab_size": 512,
        "context_window": 512,
        "weight_init": "standard",
        "head_type": "next_token",
    },
    "core_model": {
        "core_model_type": "modern",
        "hidden_dim": 512,
        "depth": 2,
        "num_heads": 8,
        "ffn_type": "swiglu",
        "normalization": "rmsnorm",
        "attn_group_size": 1,
        "ffn_dim": 512,
        "ffn_activation": "gelu",
        "dropout": 0.1,
        "bias": False,
        "attn_type": "rope",
    },
}

BYTE_TEST_CONFIG = {
    "model_shell": {
        "shell_type": "autoregressive_byte",
        "pooling_tokenizer": "BPE",
        "byte_tokenizer": "BPE",
        "tokenizer_dataset_name": "debug",
        "pooling_vocab_size": 512,
        "embedding_dim": 512,
        "vocab_size": 512,
        "context_window": 512,
        "weight_init": "standard",
        "head_type": "next_token",
    },
    "core_model": {
        "core_model_type": "modern",
        "hidden_dim": 512,
        "depth": 2,
        "num_heads": 8,
        "ffn_type": "swiglu",
        "normalization": "rmsnorm",
        "attn_group_size": 1,
        "ffn_dim": 512,
        "ffn_activation": "gelu",
        "dropout": 0.1,
        "bias": False,
        "attn_type": "rope",
    },
}

INPUT_STRING = "This is a test string to check the byte level model shell."


def test_autoregressive_model_shell():
    """Test the autoregressive model shell."""
    core_model = GenericTransformer(AUTO_TEST_CONFIG)
    model = build_shell(AUTO_TEST_CONFIG, core_model)
    assert isinstance(model, AutoregressiveModelShell)
    # test forward pass with random input

    input_tensor = torch.randint(0, 512, (1, 512))
    output, _ = model(input_tensor)
    assert output.shape == (1, 512, 512)
    # test inference

    model.eval()
    with torch.no_grad():
        output = model.inference(INPUT_STRING)
        assert output.shape == (1, 512)


def test_autoregressive_byte_model_shell():
    """Test the autoregressive byte model shell."""
    core_model = GenericTransformer(BYTE_TEST_CONFIG)
    model = build_shell(BYTE_TEST_CONFIG, core_model)
    assert isinstance(model, AutoregressiveByteModelShell)
    # test forward pass with random input

    input_tensor = torch.randint(0, 512, (1, 512))
    output, _ = model(input_tensor)
    assert output.shape == (1, 512, 512)

    # test inference

    model.eval()
    with torch.no_grad():
        output = model.inference(INPUT_STRING)
        assert output.shape == (1, 512)
