from evals.core import BaseEvaluator, BaseModelWrapper
from typing import Optional, Callable, Dict, Any
from tqdm import tqdm
import torch

# Define the metrics and their computations
METRIC_EVALUATIONS = {
    "Byte Accuracy": lambda results_dict: (
        results_dict["total_correct_bytes"] / results_dict["total_bytes"]
        if results_dict["total_bytes"] > 0 else 0.0
    ),
    "Byte Perplexity": lambda results_dict: (
        torch.exp(torch.tensor(results_dict["total_loss"] / results_dict["total_tokens"])).item()
        if results_dict["total_tokens"] > 0 else float('inf')
    ),
    "Byte Levenshtein": lambda results_dict: (
        results_dict["total_edit_distance"] / results_dict["total_bytes"]
        if results_dict["total_bytes"] > 0 else float('inf')
    ),
}

class TextModelingEvaluator(BaseEvaluator):
    """Evaluator for text modeling capabilities."""

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        yield_fn: Callable,
        yield_fn_params: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = 100,
        eval_logging_path: Optional[str] = "Text Modeling"
    ):
        super().__init__()
        self.eval_logging_path = eval_logging_path
        self.model_wrapper = model_wrapper
        self.yield_fn = yield_fn(**(yield_fn_params or {}))
        self.chunk_size = chunk_size


    def evaluate(self, model):
        """
        Evaluate the model's text modeling capabilities.

        Args:
            model: The model to be evaluated.

        Returns:
            Dict[str, float]: A dictionary with the evaluation results.
        """
        # Wrap the model using the provided model wrapper
        model = self.model_wrapper(model, chunk_size=self.chunk_size)

        results = {
            "total_bytes": 0,
            "total_correct_bytes": 0,
            "total_edit_distance": 0,
            "total_loss": 0.0,
            "total_tokens": 0
        }

        # Iterate over the reference texts
        iterator = self.yield_fn

        for reference_text in tqdm(iterator, desc="Evaluating Text Modeling"):
            # The wrapped model will return a dict with
            # bytes, correct_bytes, edit_distance, loss, tokens
            local_results = model(reference_text)
            # Expected to return:
            # {
            #     'edit_distance': total_edit_distance,
            #     'correct_bytes': total_correct_bytes,
            #     'bytes': total_bytes,
            #     'loss': total_loss,
            #     'tokens': total_tokens
            # }

            # Accumulate results
            results["total_bytes"] += local_results["bytes"]
            results["total_correct_bytes"] += local_results["correct_bytes"]
            results["total_edit_distance"] += local_results["edit_distance"]
            results["total_loss"] += local_results["loss"]
            results["total_tokens"] += local_results["tokens"]

        return {
            f"{self.eval_logging_path}/Byte Accuracy": METRIC_EVALUATIONS["Byte Accuracy"](results),
            f"{self.eval_logging_path}/Byte Perplexity": METRIC_EVALUATIONS["Byte Perplexity"](results),
            f"{self.eval_logging_path}/Byte Levenshtein": METRIC_EVALUATIONS["Byte Levenshtein"](results),
        }
