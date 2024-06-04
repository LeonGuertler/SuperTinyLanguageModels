"""Integration code"""

import torch

from models import embedding_models, generator, model_shell


def batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class EvalWrapper:
    def __init__(self, model_shell: model_shell.ModelShell):
        self.model_shell = model_shell
        super().__init__()

    def loglikelihood(self, inputs) -> list[float]:
        """
        Compute the loglikelihood of given inputs
        """
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        self.model_shell = self.model_shell.to(device)
        results = []
        with torch.no_grad():
            with torch.autocast(device_type=device_str):
                for input_strings in batch(inputs, batch_size=8):
                    batch_tokens = []
                    for input_string in input_strings:
                        embedding_model: embedding_models.EmbedderInterface = (
                            self.model_shell.embedding_model
                        )
                        # tokenize the inputs
                        input_tokens = embedding_model.tokenize_input(input_string)
                        input_tokens = embedding_model.truncate([input_tokens])[0]
                        batch_tokens.append(input_tokens)

                    padded_batch, _ = embedding_model.pad_batch(batch_tokens, direction="left")
                    input_tensor = torch.tensor(padded_batch, device=device).long()
                    # pad the input tokens to the max length in the batch
                    logits, _ = self.model_shell(input_tensor)
                    logits = logits[:, :-1, :]
                    for i,tokens in enumerate(batch_tokens):
                        ll = torch.nn.functional.cross_entropy(
                            logits[i][-len(tokens)+1:], torch.tensor(tokens[1:], device=device).long(), reduction="sum"
                        )
                        results.append(-ll.item())
        return results

    def generate(self, prefixes) -> list[str]:
        """
        Generate a continuation for a given prefix
        """
        model_generator = generator.StandardGenerator(
            self.model_shell,
            generate_cfg={
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_k": 0.9,
            },
        )
        for prefix in prefixes:
            # tokenize the inputs
            yield model_generator.default_generate(prefix)
