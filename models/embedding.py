"""
The embedding models are supposed to take care of the transformation
from text input to a sequence of tokens that can be fed into the model.
"""

import numpy as np
import tiktoken
import torch
from models.positional_encoding import LearnedPosEncoding


def build_positional_encoder(positional_encoder_type, hidden_dim, context_window):
    """
    Build the positional encoder
    Args:
        positional_encoder_type: the type of positional encoder to use
        hidden_dim: the hidden dimension of the model
        context_window: the context window of the model
    Returns:
        the positional encoder
    """
    if positional_encoder_type == "learned":
        return LearnedPosEncoding(hidden_dim=hidden_dim, context_window=context_window)
    elif positional_encoder_type == "rope":
        return None
    raise ValueError(
        f"Positional encoder type {positional_encoder_type} not recognized"
    )


class BaselineEmbedder(torch.nn.Module):
    """
    Baseline tokenizer + embedder, using GPT2 tokenizer and embeddings.
    """

    def __init__(
        self,
        hidden_dim,
        context_window,
        vocab_size,
        tokenizer_name,
        positional_encoder_type="learned",
    ):
        super().__init__()
        # load the tokenizer
        """self.tokenizer = build_tokenizer(
            tokenizer_name=tokenizer_name
        ) """
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # technically the gpt2 tokenizer has not pad token,
        # but when adjusting the attention_mask, this should
        # not matter
        self.pad_token = 1

        self.context_window = context_window

        # initialize embedding weights
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_dim
        )
        self.positional_encoding = build_positional_encoder(
            positional_encoder_type=positional_encoder_type,
            hidden_dim=hidden_dim,
            context_window=context_window,
        )

    def forward(self, text_batch, pad_truncate=True):
        """
        Convert a batch of strings into a batch
        of sequences of embeddings.
        Args:
            text_batch: a batch of strings
        Returns:
            a batch of sequences of embeddings
        """
        # tokenize the text
        token_ids_batch, attention_mask = self.tokenize_text(
            text_batch=text_batch, pad_truncate=pad_truncate
        )

        # embed and positional encode the tokens
        return self.embed_tokens(token_ids_batch), attention_mask

    def tokenize_text(self, text_batch, pad_truncate=False):
        """
        tokenize the text input batch
        Args:
            text_input: a batch of strings (B, S_c)
        Returns:
            a batch of token ids (B, S, 1)
        """
        # if text is not batched, wrap it
        if isinstance(text_batch, str):
            text_batch = [text_batch]

        # tokenize
        token_ids = self.tokenizer.encode_batch(
            text_batch,
        )

        # pad and truncate the token ids
        if pad_truncate:
            token_ids, attention_mask = self._pad_token_batch(token_ids)
        else:
            # check if all tokens sequences have the same length
            # and are shorter than the context window
            seq_len = len(token_ids[0])
            for ids in token_ids:
                assert (
                    len(ids) == seq_len
                ), "All token sequences must have the same length"
                assert (
                    len(ids) <= self.context_window
                ), f"Cannot forward sequence of length {len(ids)}, block size is only {self.context_window}"

            # convert to tensore and init attention_mask as None
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = None

        return token_ids, attention_mask

    def _pad_token_batch(self, token_ids):
        """
        Pad the token ids to the length of the longest
        sequence in the batch and return the attention mask
        Args:
            token_ids: a batch of lists of token ids
        Returns:
            token_ids: a batch of padded token ids
            attention_mask: a batch of attention masks
        """
        sequence_length = np.minimum(
            max([len(ids) for ids in token_ids]), self.context_window
        )
        attention_mask = torch.ones(len(token_ids), sequence_length)
        for i, ids in enumerate(token_ids):
            attention_mask[i, len(ids) :] = 0
            # pad where necessary
            token_ids[i] += [self.pad_token] * (sequence_length - len(ids))
            # truncate where necessary
            if len(token_ids[i]) > sequence_length:
                token_ids[i] = token_ids[i][-sequence_length:]

        return torch.tensor(token_ids, dtype=torch.long), attention_mask

    def embed_tokens(self, token_ids):
        """
        Embed given token ids
        Args:
            token_ids: a batch/list of token ids
        Returns:
            the token embeddings
        """
        token_embeddings = self.embedding(token_ids)
        if self.positional_encoding is None:
            return token_embeddings
        pos_embeddings = self.positional_encoding(token_ids)
        return token_embeddings + pos_embeddings

    def preprocess_text(self, text):
        """
        Before training, give the option to preprocess
        the text by tokenizing it.
        Args:
            text: a string
        Returns:
            the tokenized text
        """
        return self.tokenize_text(text)

    def decode_tokens(self, token_ids):
        """
        Decode given token ids
        Args:
            token_ids: a batch/list of token ids
        Returns:
            the decoded tokens
        """
        if isinstance(token_ids, list):
            return self.tokenizer.decode_batch(token_ids)
            # return [self.tokenizer.decode(token_ids) for token_ids in token_ids] # TODO
        return self.tokenizer.decode(token_ids)
