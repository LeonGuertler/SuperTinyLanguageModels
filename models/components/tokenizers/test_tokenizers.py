"""
Pytest functions for the individual components of the different tokenizers.
"""

import pytest 

from models.components.tokenizers.GPT2TokenizerWrapper import GPT2Tokenizer
from models.components.tokenizers.BytePairEncoding import BPETokenizer

# test the GPT2tokenizer
test_string = "This is a test string. <|endoftext|>"


def test_gpt2_tokenizer():
    tokenizer = GPT2Tokenizer()
    tokens = tokenizer.encode(test_string)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_string

# test the BPETokenizer
def test_bpe_tokenizer():
    tokenizer = BPETokenizer(vocab_size=512, dataset_name="debug")
    tokens = tokenizer.encode(test_string)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_string

    # try loading the tokenizer
    tokenizer = BPETokenizer(vocab_size=512, dataset_name="debug")
    tokens = tokenizer.encode(test_string)
    decoded = tokenizer.decode(tokens)
    print(tokens)
    print(decoded)
    assert decoded == test_string

    print(tokenizer.special_tokens["<|endoftext|>"])
    #tokenizer.vocab[tokenizer.special_tokens["<|endoftext|>"]]
    
