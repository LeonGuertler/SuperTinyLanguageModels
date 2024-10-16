from evals.benchmarks.yield_functions import *

from evals.core import BaseEvaluator, BaseModelWrapper

from typing import Optional, Callable, Dict, Any

class MCQEvaluator(BaseEvaluator):
    """Evaluator for multiple-choice questions."""

    def __init__(
        self, 
        yield_fn: Callable, 
        model_wrapper: BaseModelWrapper,
        yield_fn_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        """ TODO """
        self.yield_fn = yield_fn(**yield_fn_params)
        self.model_wrapper = model_wrapper
    

    def evaluate(self, model):
        """ TODO """
        model = self.model_wrapper(model) # wrap the model

        total, correct = 0, 0
        for prefix, ground_truth, false_options in self.yield_fn:
            # Prediction logic
            loglikelihoods = model(
                prefixes=[prefix] * (len(false_options) + 1),
                continuations=[ground_truth] + false_options
            )
            if loglikelihoods.index(max(loglikelihoods)) == 0:
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0
        return {
            "benchmark_type": "MCQ",
            "benchmark_name": self.env_id,
            "results": {
                "Accuracy": accuracy
            }
        }


