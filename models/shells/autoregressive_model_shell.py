"""
The Model Shell holds the tokenizer, core-model and model head.
"""

from models.shells.shell import Shell


class AutoregressiveModelShell(Shell):
    """Code that wraps the model, embedder, and head together."""

    def __init__(
        self, tokenizer, token_embedder, lm_head, core_model, weight_init_func=None
    ):
        super().__init__(
            tokenizer=tokenizer,
            token_embedder=token_embedder,
            lm_head=lm_head,
            core_model=core_model,
            weight_init_func=weight_init_func,
        )

    def embed(self, token_ids):
        """
        Embed the token ids.
        """
        return self.token_embedder(token_ids)

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
