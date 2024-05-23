"""
A simple wrapper around the GPT2 Tokenizer to
standardize the interface for tokenization.
"""

import tiktoken
import torch

from models.components.tokenizers.base_class import Tokenizer


class GPT2Tokenizer(Tokenizer):
    """A simple wrapper around the GPT2 Tokenizer."""

    def __init__(self, **_):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token
        self.pad_token = self.tokenizer.eot_token
        self.vocab_size = self.tokenizer.max_token_value + 1

    def encode(self, text):
        """Encode a string into tokens."""
        return self.tokenizer.encode_ordinary(text)

    def encode_batch(self, texts):
        """Encode a list of strings into tokens."""
        return self.tokenizer.encode_ordinary_batch(texts)

    def pad_batch(self, token_lists):
        """Pad a list of token lists to the same length,
        and return the padded tensor, and mask tensor."""
        max_len = max(len(tokens) for tokens in token_lists)
        padded_tokens = []
        mask = []
        for tokens in token_lists:
            padded_tokens.append(tokens + [self.pad_token] * (max_len - len(tokens)))
            mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
        return torch.tensor(padded_tokens), torch.tensor(mask)

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
