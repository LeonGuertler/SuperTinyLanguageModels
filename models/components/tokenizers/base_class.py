"""Base Class for Tokenizers"""

import torch


class Tokenizer:
    """Base class for tokenizers, defines the interface for tokenizers."""

    def __init__(self, **_):
        self.eot_token = 0
        self.pad_token = 0
        self.vocab_size = None

    def encode(self, text):
        """Encode a text into tokens."""
        raise NotImplementedError

    def encode_batch(self, texts):
        """Encode a batch of texts into tokens.

        Default implementation is to loop over the texts"""
        for text in texts:
            yield self.encode(text)

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
        raise NotImplementedError

    def decode_batch(self, token_lists):
        """Decode a list of token lists into a list of strings.

        Default implementation is to loop over the token lists."""
        for tokens in token_lists:
            yield self.decode(tokens)
