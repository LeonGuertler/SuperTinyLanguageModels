"""
Evaluator class for evaluating models.
"""

import torch
import tqdm

from evals import eval_wrapper
from evals.evaluator_interface import EvaluationInterface
from evals.mcqs.load_benchmarks import load_benchmark
from evals.metrics import MCQ_METRIC_DICT


class MCQEvaluator(EvaluationInterface):
    """
    Base Evaluator class the evaluates models
    and prints/logs the results.
    """

    def __init__(self, model, num_samples=None, benchmarks=None):
        self.model = model
        self.wrapper = eval_wrapper.EvalWrapper(model)
        self.num_samples = num_samples
        self.benchmarks = benchmarks
        # make sure the model is in eval model
        self.model.eval()

    @torch.no_grad()
    def predict(self, prefix, ground_truth, false_options):
        """
        Given a prompt, use the model to predict the output
        Returns the loglikelihood of the ground truth and the options
        """
        prefixes = [prefix] * (len(false_options) + 1)
        continuations = [ground_truth] + false_options
        loglikelihoods = self.wrapper.loglikelihood(prefixes=prefixes, continuations=continuations)
        loglikelihoods = torch.tensor(loglikelihoods)
        return loglikelihoods

    def _calculate_metrics(self, confidences):
        """
        Calculate the metrics for the model
        """
        score_dict = {}

        for metric_name, metric in MCQ_METRIC_DICT.items():
            score_dict[metric_name] = metric(confidences)

        return score_dict

    def evaluate_benchmark(self, benchmark_name, num_samples=None):
        """Evaluate model performance on a specific benchmark"""
        # load the benchmark_loader
        benchmark_loader = load_benchmark(benchmark_name, split="test")
        confidences = []
        for i, (prefix, ground_truth, false_options) in tqdm.tqdm(
            enumerate(benchmark_loader)
        ):
            if num_samples is not None and i > num_samples:
                break
            loglikelihoods = self.predict(prefix, ground_truth, false_options)
            confidences.append(loglikelihoods)
        # find the maximum dimension and pad the confidences up to that dimension
        max_length = max([len(confidence) for confidence in confidences])
        for i, confidence in enumerate(confidences):
            confidences[i] = torch.nn.functional.pad(
                confidence, (0, max_length - len(confidence)), value=-1e10
            )

        score_dict = self._calculate_metrics(torch.stack(confidences))

        return score_dict

    def evaluate(self):
        """Given a list of benchmark names, load and evaluate them

        Only do so on  {num_samples} for each benchmark"""
        results = {}
        for benchmark_name in self.benchmarks:
            print(f"evalling benchmark {benchmark_name}")
            score_dict = self.evaluate_benchmark(
                benchmark_name=benchmark_name, num_samples=self.num_samples
            )
            results[benchmark_name] = score_dict

        return results

    def _pretty_print_results(self, results):
        """Pretty print the results"""
        for benchmark_name, score_dict in results.items():
            print(f"{benchmark_name}:")
            for metric_name, score in score_dict.items():
                print(f"\t{metric_name}: {score}")
