"""Defines the interface for model shells"""

from torch import nn

from models.utils import print_model_stats


class Shell(nn.Module):
    """
    A model shell.
    """

    def __init__(
        self,
        tokenizer,
        token_embedder,
        lm_head,
        core_model,
        weight_init_func=None,
        context_window=1024,
    ):
        super().__init__()

        # move to class
        self.core_model = core_model

        # build components
        self.tokenizer = tokenizer
        self.token_embedder = token_embedder
        self.lm_head = lm_head

        # report number of parameters
        print_model_stats(self)

        # weight init
        self.weight_init_func = weight_init_func
        if self.weight_init_func is not None:
            self._init_weights()

        self.context_window = context_window

    def _tie_weights(self):
        """Tie the weights of the model embeddings and the final logit layer.

        see: https://paperswithcode.com/method/weight-tying"""
        self.token_embedder.weight = self.lm_head.linear.weight

    def _init_weights(self):
        """
        Initialize the weights of the model.
        Does nothing if the weight_init_func is None - in this case
        The weights are initialized by the PyTorch default initializer.
        """
        self.apply(self.weight_init_func)

    def embed(self, token_ids):
        """
        Embed the token ids.
        """
        return self.token_embedder(token_ids)

    def forward(self, token_ids):
        """
        The default forward pass is used for training and accepts the
        token_ids as input. When the model is in eval mode, only the
        last token is passed into the NextTokenHead.
        """

        _, s = token_ids.size()

        # check that the sequence length is not longer than the context window
        assert s <= self.context_window, (
            f"Cannot forward sequence of length {s}, "
            f"max window size is only {self.context_window}"
        )

        # embed token_ids
        x = self.embed(token_ids)

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
        Inference pass.
        """
        raise NotImplementedError
