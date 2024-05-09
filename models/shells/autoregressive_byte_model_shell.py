"""
The Model Shell holds the tokenizer, core-model and model head.
"""

import torch
from torch import nn

from models.components.layers import BidirectionalTransformerBlock
from models.shells.shell import Shell


class AutoregressiveByteModelShell(Shell):
    """
    A model shell for byte-level learning.
    """

    def embed(self, token_ids):
        """
        Embed the token ids.
        """
        return self.tokenizer(token_ids)

    def inference(self, sequence):
        """
        Similar to the forward pass, but takes in a string
        (or batch of strings) and only return the logits
        for the next token.
        Args:
            text_string: a string or list of strings
        Returns:
            logits for the next token
        """
        if isinstance(sequence, str):
            sequence = [sequence]

        tokens = self.tokenizer.pooling_tokenizer.encode_batch(sequence)
        tokens = torch.tensor(tokens)

        _, s = tokens.size()

        # check that the sequence length is not longer than the context window
        assert s <= self.context_window, (
            f"Cannot forward sequence of length {s},"
            f"max window size is only {self.context_window}"
        )

        # process to sub-word tokens
        x = self.tokenizer(tokens)

        # forward through the core model
        x_return = self.core_model(x)
        if isinstance(x, tuple):
            x, _ = x_return
        else:
            x, _ = x_return, None

        # get logits
        logits = self.lm_head(x)

        return logits[:, -1, :]


class ByteLevelProcessor(nn.Module):
    """
    Takes byte level encodings, processes them via
    two local-attention transformer blocks and pools
    the resultant tokens based on gpt-2 tokenizer
    boundaries.
    Inputs are batches of lists of token blocks
    in the gpt2 tokenizer boundaries.
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        pooling_tokenizer,
        byte_tokenizer,
        token_embedder,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pooling_tokenizer = pooling_tokenizer
        self.byte_tokenizer = byte_tokenizer
        self.token_embedder = token_embedder
        self.transformer = nn.ModuleList(
            [
                BidirectionalTransformerBlock(
                    hidden_dim=embedding_dim,
                    ffn_dim=embedding_dim * 4,
                    ffn_activation="gelu",
                    bias=False,
                    num_heads=8,
                ),
                BidirectionalTransformerBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=hidden_dim * 4,
                    ffn_activation="gelu",
                    bias=False,
                    num_heads=8,
                ),
            ]
        )
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def token_to_byte_ids(self, token_ids):
        """
        Convert token ids to byte ids

        Token_ids: of shape (batch_size, sequence_length)
        """
        token_strings = self.pooling_tokenizer.decode_batch(token_ids)
        byte_ids = []
        for token_string in token_strings:
            byte_id = self.byte_tokenizer.encode(token_string)
            byte_ids.append(byte_id)
        return byte_ids

    def forward(self, batch_of_pooled_token_ids):
        """
        A batch of lists of tensors (token ids)
        """

        full_batch = torch.zeros(
            (
                batch_of_pooled_token_ids.size(0),
                batch_of_pooled_token_ids.size(1),
                12,
                self.embedding_dim,
            ),
        )

        for i, token_batch in enumerate(batch_of_pooled_token_ids):
            # iterate over actual ids

            for j, token_id in enumerate(token_batch):
                # decode into string
                token_string = self.pooling_tokenizer.decode([token_id.item()])
                # encode into character ids
                byte_ids = self.byte_tokenizer.encode(token_string)
                # convert to tensor
                byte_ids = torch.tensor(byte_ids)
                num_ids = len(byte_ids)
                if num_ids > 12:
                    byte_ids = byte_ids[:12]
                    num_ids = 12

                # embed
                x = self.token_embedder(byte_ids).unsqueeze(0)

                # add to full batch
                full_batch[i, j, :num_ids] = x

        # print(full_batch.size())
        B, S, char_seq_len, E = full_batch.size()
        full_batch = full_batch.view(B * S, char_seq_len, E)
        full_batch = self.transformer[0](full_batch)
        full_batch = self.up_proj(full_batch)
        full_batch = self.transformer[1](full_batch)
        full_batch = full_batch.mean(dim=-2)
        full_batch = full_batch.view(B, S, -1)
        return full_batch
