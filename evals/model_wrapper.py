"""Model wrapper for use on benchmarks"""

from contextlib import nullcontext
import Levenshtein
import torch


class ModelWrapper:
    """Wrapper for running models on benchmarks"""

    def __init__(self, model):
        self.model = model
        self.ctx = nullcontext()

    def predict(self, prompts, options):
        """Predicts the best option for each prompt"""
        outputs = []
        with self.ctx:
            with torch.no_grad():
                outputs = [self.model.default_generate(prompt) for prompt in prompts]
        predictions = []
        for output, option in zip(outputs, options):
            best, best_score = None, float("inf")
            for opt in option:
                score = Levenshtein.distance(output, opt)
                if score < best_score:
                    best, best_score = opt, score
            predictions.append(best)
        return predictions

    def encode(self, _):
        """Embed the input"""
        pass
