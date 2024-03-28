"""Model wrapper for use on benchmarks"""

import Levenshtein
import torch


class ModelWrapper:
    def __init__(self, model, ctx, cfg):
        self.model = model
        self.ctx = ctx
        self.generate_config = cfg

    def predict(self, prompts, options):

        outputs = []
        with self.ctx:
            with torch.no_grad():
                outputs = [
                    self.model.generate(
                        prompt,
                        self.generate_config["max_new_tokens"],
                        self.generate_config["temperature"],
                        self.generate_config["top_k"],
                    )
                    for prompt in prompts
                ]
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
