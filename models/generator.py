"""
Generator Base Wrapper
"""

import torch


class StandardGenerator(torch.nn.Module):
    """Standard Generator Wrapper for GPT models"""

    def __init__(self, model, generate_cfg):
        """Initialize the model and the configuration"""
        super().__init__()
        self.model = model
        self.model = self.model.to(torch.device("cuda"))
        self.generate_config = generate_cfg

    def default_generate(self, input_text):
        """
        Generate text using the default generation method
        """
        return self.generate(
            input_text,
            self.generate_config["max_new_tokens"],
            self.generate_config["temperature"],
            self.generate_config["top_k"],
        )

    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = self.model.embedding_model.tokenize_input(input_string=input_text)
        # push to device
        idx = torch.tensor(idx).unsqueeze(0).to(torch.device("cuda"))
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.model.inference(idx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # logits have shape (b,t,v)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # check for dim
                if len(v.size()) == 3:
                    logits[logits < v[:, :, [-1]]] = -float("Inf")
                else:
                    logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            # check if byte-level and if so, flatten
            if len(probs.size()) == 3:
                B, S, H = probs.size()
                probs = probs.view(B * S, H)
                flattened = True
            else:
                flattened = False

            idx_next = torch.multinomial(probs, num_samples=1)

            # check if byte-level and if so, unflatten
            if flattened:
                idx_next = idx_next.view(B, S)
            elif idx_next == self.model.embedding_model.eot_token:
                break

            if flattened:
                idx_next = idx_next.unsqueeze(0)
            idx = torch.cat((idx, idx_next), dim=1)

        return self.model.embedding_model.decode(idx.tolist())

    def forward(self, x):
        """Call the underlying model"""
        return self.model(x)

    def embed(self, x):
        """Embed the input"""
        return self.model.embed(x)


def build_generator(model, generate_cfg):
    """Build the generator"""
    return StandardGenerator(model, generate_cfg)
