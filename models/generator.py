"""Generator Base Wrapper"""

import models.baseline as baseline
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardGenerator(nn.Module):
    """Standard Generator Wrapper for GPT models"""

    def __init__(self, model: baseline.BaseGPT, generate_cfg):
        """Initialize the model and the configuration"""
        super().__init__()
        self.model = model
        self.generate_config = generate_cfg

    def default_generate(self, input_text):
        """
        Generate text using the default generation method
        """
        return self.generate(
            input_text,
            self.generate_config["max_new_tokens"],
            self.generate_config["temperature"],
            self.generate_config["top_k"]
        )

    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        input_string = input_text
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.model.inference(input_string)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next == self.model.embedder.tokenizer.eot_token:
                break
            new_char = self.model.embedder.tokenizer.decode([idx_next])

            input_string += new_char
        return input_string[len(input_text):]

    def forward(self, x):
        """Call the underlying model"""
        return self.model(x)

    def embed(self, x):
        """Embed the input"""
        return self.model.embed(x)


def build_generator(model, generate_cfg):
    """Build the generator"""
    return StandardGenerator(model, generate_cfg)
