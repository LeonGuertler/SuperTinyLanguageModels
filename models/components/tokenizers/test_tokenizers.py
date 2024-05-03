"""
Pytest functions for the individual components of the different tokenizers.
"""

# pylint: disable=unused-import
# pylint: disable=missing-function-docstring
import pytest

from models.components.tokenizers.byte_pair_encoding import BPETokenizer
from models.components.tokenizers.gpt2_tokenizer import GPT2Tokenizer

# test the GPT2tokenizer
TEST_STRING = "This is a test string. <|endoftext|>"


def test_gpt2_tokenizer():
    tokenizer = GPT2Tokenizer()
    tokens = tokenizer.encode(TEST_STRING)
    decoded = tokenizer.decode(tokens)
    assert decoded == TEST_STRING


def test_bpe_tokenizer():
    tokenizer = BPETokenizer(vocab_size=512, dataset_name="debug")
    tokens = tokenizer.encode(TEST_STRING)
    decoded = tokenizer.decode(tokens)
    assert decoded == TEST_STRING

    # try loading the tokenizer
    tokenizer = BPETokenizer(vocab_size=512, dataset_name="debug")
    tokens = tokenizer.encode(TEST_STRING)
    decoded = tokenizer.decode(tokens)
    print(tokens)
    print(decoded)
    assert decoded == TEST_STRING

    print(tokenizer.special_tokens["<|endoftext|>"])
    # tokenizer.vocab[tokenizer.special_tokens["<|endoftext|>"]]
