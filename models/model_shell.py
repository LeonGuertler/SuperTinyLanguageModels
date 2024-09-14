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
        model_cfg,
        embedding_model: embedding_models.EmbedderInterface,
        core_model: core_models.GenericTransformer,
        model_head: model_heads.AutoregressiveLMHead,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head


        # check if embedding model weights are to be shared with the model head
        if model_cfg.get("embedding_weight_tying", True):
            # share the weights between the token embeddings and the final
            # logit layer, following: https://paperswithcode.com/method/weight-tying
            assert model_head.linear.weight.shape == embedding_model.token_embedder.weight.shape, \
                "The embedding model and the model head should have the same output dimension."
            embedding_model.token_embedder.weight = model_head.linear.weight

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
    def loglikelihood(self, prefixes, continuations):
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
        logits = logits[:, :-1].reshape(-1, logits.size(-1))
        target_tensor = input_tensor[:, 1:].reshape(-1)
        ll = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
        mask = mask[:, 1:].reshape(-1).to(ll.device)
        ll = ll * mask
        ll = ll.view(input_tensor.size(0), -1).sum(dim=1)
        return -ll


    @torch.no_grad()
    def generate(self, 
        prompt: str, 
        max_new_tokens: int=100, 
        temperature: float=1.0, 
        top_k: int=None, 
        repetition_penalty: float=None, 
        repetition_window: int=None
    ):
        """ Basic text generation function. """

        # tokenize input tokens
        idx = self.embedding_model.tokenize_input(
            input_string=prompt,
            add_eot=False,
            truncate=True
        )

        # push to device
        idx = torch.tensor(idx).unsqueeze(0).to(torch.device(self.device))
        for _ in range(max_new_tokens):
            # forward the model to get the logits for th index in the sequence
            logits, _ = self.inference(idx)

            # scale by temp
            logits = logits / temperature


            # apply repetition penalty
            if repetition_penalty is not None:
                # Get the most recent tokens within the window
                recent_tokens = idx[0, -repetition_window:]
                # Count the occurrences of each token
                unique_tokens, counts = torch.unique(
                    recent_tokens,
                    return_counts=True
                )
                # Apply penalty to the logits of repeated tokens
                logits[0, unique_tokens] /= repetition_penalty ** counts.float()


            # crop logits to top_k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")


            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the disstribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # check if end of text
            if idx_next == self.embedding_model.eot_token:
                break 

            # otherwaise append
            idx = torch.cat((idx, idx_next), dim=1)

        # return back as string
        return self.embedding_model.decode(idx.tolist())
