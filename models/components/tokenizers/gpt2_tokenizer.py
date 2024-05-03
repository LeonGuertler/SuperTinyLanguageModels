"""
A simple wrapper around the GPT2 Tokenizer to
standardize the interface for tokenization.
"""

import tiktoken


class GPT2Tokenizer:
    """A simple wrapper around the GPT2 Tokenizer."""

    def __init__(self, **_):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token

    def encode(self, text):
        """Encode a string into tokens."""
        return self.tokenizer.encode_ordinary(text)

    def decode(self, tokens):
        """Decode a list of tokens into a string."""
        return self.tokenizer.decode(tokens)
