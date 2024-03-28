"""Model wrapper for use on benchmarks"""

import Levenshtein
import torch


class ModelWrapper:
    def __init__(self, model, ctx):
        self.model = model
        self.ctx = ctx

    def predict(self, prompts, options):

        outputs = []
        with self.ctx:
            with torch.no_grad():
                output = self.model.generate(
                    prompts,
                    options["max_new_tokens"],
                    options["temperature"],
                    options["top_k"],
                )
        for output, option in zip(outputs, options):
            best, best_score = None, float("inf")
            for opt in option:
                score = Levenshtein.distance(output, opt)
                if score < best_score:
                    best, best_score = opt, score
        outputs.append(best)

        return outputs

    def embed(self, text):
        return self.model()
