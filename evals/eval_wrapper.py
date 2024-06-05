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

    def loglikelihood(self, prefixes, continuations) -> list[float]:
        """
        Compute the loglikelihood of given inputs
        """
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        self.model_shell = self.model_shell.to(device)
        results = []
        with torch.no_grad():
            with torch.autocast(device_type=device_str):
                for prefix_batch, cont_batch in zip(
                    batch(prefixes, 32), batch(continuations, 32)
                ):
                    ll = self.model_shell.loglikelihood(prefix_batch, cont_batch)
                    results.extend(ll.cpu().numpy())
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
