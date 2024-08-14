"""
A simple wrapper around the GPT2 Tokenizer to
standardize the interface for tokenization.
"""

import tiktoken
import torch

from models.components.tokenizers.base_class import Tokenizer


class P50KTokenizer(Tokenizer):
    """A simple wrapper around the GPT2 Tokenizer."""

    def __init__(self, **_):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("p50k_base")
        self.eot_token = self.tokenizer.eot_token
        self.pad_token = self.tokenizer.eot_token
        self.vocab_size = self.tokenizer.max_token_value + 1

    def encode(self, text):
        """Encode a string into tokens."""
        return self.tokenizer.encode_ordinary(text)

    def encode_batch(self, texts):
        """Encode a list of strings into tokens."""
        return self.tokenizer.encode_ordinary_batch(texts)

    def decode(self, tokens):
        """Decode a list of tokens into a string."""
        # check if the tokens are a tensor
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def decode_batch(self, token_lists):
        """Decode a list of token lists into a list of strings."""
        if torch.is_tensor(token_lists):
            token_lists = token_lists.tolist()
        return self.tokenizer.decode_batch(token_lists)
