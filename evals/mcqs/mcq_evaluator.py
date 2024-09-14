"""
Evaluator class for evaluating the models ability to answer
different mcq style questions correctly.
"""

import torch
import tqdm

from evals import eval_wrapper
from evals.evaluator_interface import EvaluationInterface
from evals.mcqs.load_benchmarks import load_benchmark 
from trainers.utils import aggregate_value

class MCQEvaluator(EvaluationInterface):
    """
    Base Evaluator class the evaluates models
    and prints/logs the results.
    """

    def __init__(self, model, num_samples=None, benchmark_list=None):
        self.model = model
        self.wrapper = eval_wrapper.EvalWrapper(model)
        self.num_samples = num_samples
        self.benchmark_list = benchmark_list
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

    def evaluate_benchmark(self, benchmark_name, num_samples=None):
        """Evaluate model performance on a specific benchmark"""
        # load the benchmark_loader
        benchmark_loader = load_benchmark(
            benchmark_name=benchmark_name, 
            num_samples=num_samples
        )
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

        # Stack the tensor values to form a single 2D tensor
        confidences = torch.stack(confidences)

        # calculate the accuracy and return it
        _, predicted = torch.max(confidences, 1)
        # aggregate the tensor values
        return aggregate_value((predicted == 0).float().mean())

    def evaluate(self):
        """Given a list of benchmark names, load and evaluate them

        Only do so on  {num_samples} for each benchmark"""
        results = {}
        for benchmark_name in self.benchmark_list:
            print(f"evalling benchmark {benchmark_name}")
            accuracy = self.evaluate_benchmark(
                benchmark_name=benchmark_name, num_samples=self.num_samples
            )
            results[f"MCQ/{benchmark_name}"] = accuracy

        return results
