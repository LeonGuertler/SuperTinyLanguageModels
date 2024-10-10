"""
A collection of embedding models. A collection model includes
the tokenizer(s), token embeddings and positional encodings
(if necessary).
"""

import torch
import numpy as np 

from models.components.positional_encoding import build_positional_encodings
from models.components.tokenizers import build_tokenizer


class EmbedderInterface(torch.nn.Module):
    """Interface for the embedder component of the model."""

    def __init__(self):
        super().__init__()
        self.eot_token = ...

    def forward(self, token_ids: torch.LongTensor):
        """This function should take the token_ids as input,

        and return the embeddings."""
        raise NotImplementedError

    def tokenize_input(self, input_string: str, truncate=False, add_eot=True):
        """This function should take a single input string and returns

        the tokenized input.
        Args:
            input_string: str
            truncate: bool - whether to perform (left) truncation
            add_eot: bool
        Returns:
            typically token_ids of shape (S,)
        """
        raise NotImplementedError

    def decode(self, tokens: torch.LongTensor):
        """This function should decode a tensor of tokens into a string.

        For the default implementation of get_sequence_info,
        we assume that the tokens are of shape (B, S) and we
        decode each sequence in the batch."""
        raise NotImplementedError

    def inference(self, input_string: str, add_eot=False):
        """This function should map string to embeddings."""
        token_ids = self.tokenize_input(input_string, truncate=True, add_eot=add_eot)
        token_ids = torch.tensor(token_ids).unsqueeze(0).to(
            next(self.parameters()).device
        )
        return self.forward(token_ids)

    def pad_batch(self, token_lists, direction="right"):
        """Pad a list of token lists to the same length,
        and return the padded tensor, and mask tensor."""
        raise NotImplementedError

    def truncate(self, token_lists):
        """Truncate a list of token lists, to be shorter than the,
        maximum length of the model and return the truncated tensor.
        """
        raise NotImplementedError

    def get_sequence_info(self, x):
        """
        Given a batch of sequences of tokens, return
        the character lengths.
        Args:
            x: torch.tensor(B, S)
        """

        sequence_char_lengths = []
        # then we decode everything
        # batch decode
        sequences = self.tokenizer.decode_batch(x)
        for seq in sequences:
            sequence_char_lengths.append(len(seq))

        # obtain the mask for end-of-word and pad tokens
        mask = x != self.tokenizer.pad_token
        mask = mask & (x != self.tokenizer.eot_token)

        return (
            sequence_char_lengths,
            mask,
        )


class GenericEmbedder(EmbedderInterface):
    """
    A simple and flexible embedding model.

    All embedders should inherit from this class.
    """

    def __init__(self, model_cfg):
        super().__init__()
        # build the tokenizer
        self.tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["tokenizer_type"],
            vocab_size=model_cfg.get("vocab_size", None),
            dataset_name=model_cfg.get("tokenizer_dataset_name", None),
            simplify=model_cfg.get("tokenizer_simplify", True), # Default True
            num_reserved_tokens=model_cfg.get("tokenizer_num_reserved_tokens", 0), # By Default, no spaces are reserved
        )

        # build the token embeddings
        self.token_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["hidden_dim"],
        )

        # build the positional encodings
        self.positional_encodings = build_positional_encodings(model_cfg=model_cfg)
        self.eot_token = self.tokenizer.eot_token
        self.model_cfg = model_cfg

        self.dropout = torch.nn.Dropout(p=model_cfg.get("embedding_dropout", 0.0))

        self.token_byte_length_cache = self._precompute_byte_lengths()

    def _precompute_byte_lengths(self):
        """
        Precompute byte lengths for all tokens in the vocabulary.
        """
        vocab_size = self.tokenizer.vocab_size
        token_byte_lengths = np.zeros(vocab_size, dtype=np.int32)
        for token in range(vocab_size):
            token_str = self.tokenizer.decode([token])
            token_bytes = token_str.encode('utf-8')
            token_byte_lengths[token] = len(token_bytes)
        return token_byte_lengths

    def get_byte_lengths(self, tokens):
        """
        Given a list/array of tokens, return a NumPy array of byte lengths using the cache.
        """
        tokens = np.array(tokens)
        byte_lengths = self.token_byte_length_cache[tokens]
        return byte_lengths

    def forward(self, token_ids):
        """
        Takes the token_ids as input
        and returns the embeddings.

        To obtain the token ids, use `.tokenize_input()`
        Args:
            token_ids: torch.tensor(B, S)
        Returns:
            embeddings: torch.tensor(B, S, H)
        """

        # get the token embeddings
        x = self.token_embedder(token_ids)

        # apply the positional encoding, if any
        x = self.positional_encodings(x)

        return x

    def tokenize_input(self, input_string, truncate=False, add_eot=True):
        """
        Tokenize an input string.
        """
        token_ids = self.tokenizer.encode(input_string)
        if add_eot:
            token_ids.append(self.eot_token)
        if truncate:
            token_ids = self.truncate([token_ids])[0]
        return token_ids

    def pad_batch(self, token_lists, direction="right"):
        """Pad a list of token lists to the same length,
        and return the padded tensor, and mask tensor.
        Args:
            token_lists: list of lists of tokens
            direction: str
        """
        return self.tokenizer.pad_batch(token_lists, direction=direction)

    def truncate(self, token_lists):
        # get model max length
        max_length = self.model_cfg["context_window"]
        return [token_seq[-max_length:] for token_seq in token_lists]

    def decode(self, tokens):
        """
        Decode a tensor of tokens into a string.
        """
        return self.tokenizer.decode_batch(tokens)
