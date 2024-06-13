"""
The standard Model Shell. It combines the embedding model,
core model and LM head.
"""

import torch

from models import core_models, embedding_models, model_heads



class ModelShell(torch.nn.Module):
    """
    Unify the embedding model, core model and LM head
    into a single object; initializes the weights
    and prints basic model statistics.
    """

    def __init__(
        self,
        embedding_model: embedding_models.EmbedderInterface,
        core_model: core_models.GenericTransformer,
        model_head: model_heads.AutoregressiveLMHead,
        weight_init_func=None,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head

        # initialize model weights
        if weight_init_func is not None:
            self.apply(weight_init_func)
        self.device = ...

    # override to device to set the attribute
    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(self, token_ids):
        """
        The default forward pass is used for trianing and
        accepts the token_ids as input.
        """

        # pass the token_ids through the embedding model
        # to get B, S, H (with pos encoding if necessary)
        x = self.embedding_model(token_ids)

        # pass the embeddings through the core model
        x = self.core_model(x)

        # pass the core model output through the model head
        x = self.model_head(x)

        return x

    @torch.no_grad()
    def inference(self, model_input):
        """
        Takes a string or list of token ids as input,
        and returns the decoded model output. The actual
        decoding should happen in the decoding generator.
        Args:
            model_input: str or torch.tensor(B, S)
        Returns:
            logits: torch.tensor(B, S, V),
        """

        # check if input is string
        if isinstance(model_input, str):
            # use inference function of the embedding model
            model_input = self.embedding_model.tokenize_input(model_input, truncate=True, add_eot=False)
        x = torch.tensor(model_input, device=self.device, dtype=torch.long).unsqueeze(0)
        x = self.embedding_model(model_input)

        # pass the embeddings through the core model
        x = self.core_model(x)

        # pass the core model output through the model head
        logits = self.model_head.inference(x)

        return logits, model_input

    @torch.no_grad()
    def continuation_likelihood(self, prefixes, continuations):
        """
        Compute the loglikelihood of continuation
        tokens given a prefix.
        Args:
            prefixes: list[str]
            continuations: list[str]
        Returns:
            ll: torch.tensor(B)
        """
        total_strings = [f"{prefix} {cont}" for prefix, cont in zip(prefixes, continuations)]
        input_tokens = [self.embedding_model.tokenize_input(string, truncate=True) for string in total_strings]
        padded_batch, mask = self.embedding_model.pad_batch(input_tokens, direction="right")
        input_tensor = torch.tensor(padded_batch, device=self.device, dtype=torch.long)
        logits, _ = self.forward(input_tensor)
        logits = logits[:, :-1]
        target_tensor = input_tensor[:, 1:]
        mask = mask[:, 1:].to(logits.device)
        return self.loglikelihood(logits, target_tensor, mask)

    def loglikelihood(self, logits, target_tensor, mask):
        """Given output logits and the target tensor for them,
        
        Compute the path probability of the target tensor.
        
        Arguments:
            logits {torch.tensor} -- Logits of the model output
            target_tensor {torch.tensor} -- Target tensor
                these should be right shifted by one...
            mask {torch.tensor} -- Mask for the target tensor"""
        # reshape the tensors
        B, S, V = logits.size()
        logits = logits.reshape(B * S, V)
        target_tensor = target_tensor.reshape(-1)
        mask = mask.reshape(-1)
        # compute the log likelihood of the target_tensor
        ll = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
        # apply the mask
        ll = ll * mask
        # bring back the batch dim...
        ll = ll.reshape(B, -1)
        return -ll.sum(dim=1)