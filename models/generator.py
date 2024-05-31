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
        idx = self.model.embedding_model.tokenize_input(
            input_string=input_text
        )
        # push to device
        idx = torch.tensor(idx).unsqueeze(0).to(torch.device("cuda"))
        # input_string = input_text
        # output_tokens = []
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.model.inference(idx)
            input(logits.size())
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                input(v.size())
                # check for dim
                if len(v.size()) == 3:
                    logits[logits < v[:, [-1]]] = -float("Inf")
                else:
                    logits[logits < v[:, :, :, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            # check if byte-level and if so, flatten 
            if len(probs.size()) == 4:
                B, S, S_c, H = probs.size()
                probs = probs.view(B* S * S_c, H)
                flattened = True
            else:
                flattened = False


            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # check if byte-level and if so, unflatten
            if flattened:
                idx_next = idx_next.view(B, S, S_c)


            if idx_next == self.model.tokenizer.eot_token:
                break


            # new_char = self.model.tokenizer.decode([idx_next.item()])
            idx = torch.cat((idx, idx_next), dim=1)
            # output_tokens.append(idx_next.item())
            # input_string += new_char

        return self.model.embedding_model.decode(idx[0].tolist())
        #return self.tokenizer.decode_tokens(idx[0].tolist())
        # return self.model.tokenizer.decode(output_tokens)
        # return input_string[len(input_text):]

    def forward(self, x):
        """Call the underlying model"""
        return self.model(x)

    def embed(self, x):
        """Embed the input"""
        return self.model.embed(x)


def build_generator(model, generate_cfg):
    """Build the generator"""
    return StandardGenerator(model, generate_cfg)