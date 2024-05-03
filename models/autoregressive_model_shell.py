"""
The Model Shell holds the tokenizer, core-model and model head.
"""

import torch
from torch import nn

from models.components.lm_heads import NextTokenHead
from models.components.tokenizers import build_tokenizer
from models.utils import print_model_stats
from models.weight_initialization import build_weight_init


class AutoregressiveModelShell(nn.Module):
    """Code that wraps the model, embedder, and head together."""

    def __init__(
        self,
        cfg,
        core_model,
    ):
        super().__init__()

        # move to class
        self.cfg = cfg
        self.core_model = core_model

        # build the tokenizer
        self.tokenizer = build_tokenizer(
            tokenizer_type=self.cfg["model_shell"]["tokenizer"],
            vocab_size=self.cfg["model_shell"]["vocab_size"],
            dataset_name=self.cfg["model_shell"]["tokenizer_dataset_name"],
        )

        # build the embedder
        self.token_embedder = nn.Embedding(
            num_embeddings=self.cfg["model_shell"]["vocab_size"],
            embedding_dim=self.cfg["core_model"]["hidden_dim"],
        )

        # build the language model head
        self.lm_head = NextTokenHead(
            hidden_dim=self.cfg["core_model"]["hidden_dim"],
            vocab_size=self.cfg["model_shell"]["vocab_size"],
        )

        # share the weights between the token embeddings and the final logit layer
        self.token_embedder.weight = (
            self.lm_head.linear.weight
        )  # https://paperswithcode.com/method/weight-tying

        # report number of parameters
        print_model_stats(self)

        # weight init
        self.weight_init_func = build_weight_init(
            weight_init_type=self.cfg["model_shell"]["weight_init"],
        )
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model
        """
        self.apply(self.weight_init_func)

    def forward(self, token_ids):
        """
        The default forward pass is used for training and accepts the
        token_ids as input. When the model is in eval mode, only the
        last token is passed into the NextTokenHead.
        """

        _, s = token_ids.size()

        # check that the sequence length is not longer than the context window
        assert s <= self.cfg["model_shell"]["context_window"], (
            f"Cannot forward sequence of length {s}, "
            f"block size is only {self.cfg['model_shell']['context_window']}"
        )

        # embed token_ids
        x = self.token_embedder(token_ids)

        # forward through the core model
        x_return = self.core_model(x)
        if isinstance(x, tuple):
            x, loss = x_return
        else:
            x, loss = x_return, None

        # get logits
        logits = self.lm_head(x)

        return logits, loss

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

        token_ids = self.tokenizer.encode_batch(sequence)

        # pad token_ids and format as tensor
        tokens, _ = self.tokenizer.pad_batch(token_ids)
        # ignore mask for now...

        _, s = tokens.size()

        # check that the sequence length is not longer than the context window
        assert s <= self.cfg["model_shell"]["context_window"], (
            f"Cannot forward sequence of length {s}, "
            f"block size is only {self.cfg['model_shell']['context_window']}"
        )

        # embed token_ids
        x = self.token_embedder(tokens)

        # forward through the core model
        x = self.core_model(x)

        # forward only the last token through the lm_head
        logits = self.lm_head(x[:, -1, :])
        return logits
