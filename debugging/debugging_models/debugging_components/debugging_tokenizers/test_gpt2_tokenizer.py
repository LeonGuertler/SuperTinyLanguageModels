"""
Pytest for the GPT2 tokenizer.
"""

import pytest
import torch

from models.components.tokenizers import build_tokenizer


def test_gpt2_tokenizer():
    """
    Build the GPT2 tokenizer and encode decode text.
    """
    tokenizer = build_tokenizer(
        tokenizer_type="gpt2", vocab_size=50257, dataset_name=None
    )

    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text

    text = "This is a test."
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text
