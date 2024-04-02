"""
The embedding models are supposed to take care of the transformation
from text input to a sequence of tokens that can be fed into the model.
"""
import torch
import torch.nn as nn

import tiktoken

from positional_encoding import (
    LearnedPosEncoding
)




class BaselineEmbedder(torch.nn.Module):
    """
    Baseline tokenizer + embedder, using GPT2 tokenizer and embeddings.
    """
    def __init__(
            self,
            hidden_dim,
            context_window,
            batch_size,
        ):
        super().__init__()
        # load the tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.context_window = context_window
        self.batch_size = batch_size


        # initialize embedding weights
        self.embedding = torch.nn.Embedding(
            self.tokenizer.vocab_size,
            hidden_dim
        )
        self.positional_encoding = LearnedPosEncoding(
            hidden_dim=hidden_dim,
            context_window=context_window
        )


    def forward(self, text_batch):
        """
        Convert a batch of strings into a batch 
        of sequences of embeddings.
        Args:
            text_batch: a batch of strings
        Returns:
            a batch of sequences of embeddings
        """
        # if text is not batched, wrap it
        if isinstance(text_batch, str):
            text_batch = [text_batch]

        # tokenize the text
        token_ids_batch = self.tokenize_text(text_batch)
        
        # check if it is within the legal context window
        assert (
            token_ids_batch.size(1) <= self.context_window
        ), f"Cannot forward sequence of length {token_ids_batch.size(1)}, block size is only {self.context_window}"

        # embed and positional encode the tokens
        return self.embed_tokens(token_ids_batch)
    

    def tokenize_text(self, text_input, truncate=True):
        """
        tokenize the text input batch
        Args:
            text_input: a batch of strings (B, S_c)
        Returns:
            a batch of token ids (B, S, 1)
        """
        # tokenize
        token_ids = self.tokenizer.encode_batch(
            text_input,
            pad_to_max_length=True
        )

        if truncate:
            token_ids = token_ids[:, -self.context_window:]
        else:
            # assert if token ids are within the context window
            assert (
                    token_ids.size(1) <= self.context_window
                ), f"Cannot forward sequence of length {token_ids.size(0)}, block size is only {self.context_window}, set truncate=True to truncate the sequence."
        return token_ids


    def embed_tokens(self, token_ids):
        """
        Embed given token ids
        Args:
            token_ids: a batch/list of token ids
        Returns:
            the token embeddings
        """
        token_embeddings = self.embedding(token_ids)
        pos_embeddings = self.positional_encoding(token_embeddings)
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
            #return [self.tokenizer.decode(token_ids) for token_ids in token_ids] # TODO
        return self.tokenizer.decode(token_ids)
    


 