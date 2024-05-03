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
TEST_STRING_2 = "This is a test string."
TEST_STRING_3 = "test"


def test_gpt2_tokenizer():
    tokenizer = GPT2Tokenizer()
    tokens = tokenizer.encode(TEST_STRING)
    decoded = tokenizer.decode(tokens)
    assert decoded == TEST_STRING

    tokens_list = tokenizer.encode_batch([TEST_STRING, TEST_STRING_2, TEST_STRING_3])
    decoded_list = tokenizer.decode_batch(tokens_list)
    assert decoded_list[0] == TEST_STRING
    assert decoded_list[1] == TEST_STRING_2

    tensor, mask = tokenizer.pad_batch(tokens_list)
    assert tensor.shape == mask.shape


def test_bpe_tokenizer():
    tokenizer = BPETokenizer(vocab_size=512, dataset_name="debug")
    tokens = tokenizer.encode(TEST_STRING)
    decoded = tokenizer.decode(tokens)
    assert decoded == TEST_STRING

    # try loading the tokenizer
    tokenizer = BPETokenizer(vocab_size=512, dataset_name="debug")
    tokens = tokenizer.encode(TEST_STRING)
    decoded = tokenizer.decode(tokens)
    assert decoded == TEST_STRING

    tokens_list = tokenizer.encode_batch([TEST_STRING, TEST_STRING_2, TEST_STRING_3])
    decoded_list = tokenizer.decode_batch(tokens_list)
    assert decoded_list[0] == TEST_STRING
    assert decoded_list[1] == TEST_STRING_2

    tensor, mask = tokenizer.pad_batch(tokens_list)
    assert tensor.shape == mask.shape
    # tokenizer.vocab[tokenizer.special_tokens["<|endoftext|>"]]
