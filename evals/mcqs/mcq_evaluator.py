"""
Evaluator class for evaluating models.
"""

import torch

from evals.mcqs.load_benchmarks import load_benchmark
from evals.metrics import MCQ_METRIC_DICT


class MCQEvaluator:
    """
    Base Evaluator class the evaluates models
    and prints/logs the results.
    """

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

        # make sure the model is in eval model
        self.model.eval()

    @torch.no_grad()
    def predict(self, prompt_list, options_list=None):
        """
        Given a prompt, use the model to predict the output
        (if necessary, restrict the output space to the tokens
        given in the options list)
        """
        iterator = (
            zip(prompt_list, [None] * len(prompt_list))
            if options_list is None
            else zip(prompt_list, options_list)
        )
        answer_list = []
        for prompt, options in iterator:
            # predict the next token
            output = self.model.inference(prompt)

            # restrict the output space to options
            if options is not None:
                legal_idx = self.model.embedder.tokenizer.encode(options)
                output = output[legal_idx]
                probs = torch.softmax(output, dim=-1)
                output = torch.multinomial(probs, num_samples=1)
                answer = options[output]
            else:
                # just sample the next token
                output = torch.multinomial(output, num_samples=1)
                answer = self.model.embedder.tokenizer.decode(output)

            answer_list.append(answer)

        return answer_list

    def _calculate_metrics(self, predictions, targets):
        """
        Calculate the metrics for the model
        """
        score_dict = {}

        for metric_name, metric in MCQ_METRIC_DICT.items():
            score_dict[metric_name] = metric(predictions=predictions, targets=targets)

        return score_dict

    def evaluate_benchmark(self, benchmark_name):
        """Evaluate model performance on a specific benchmark"""
        # load the benchmark_loader
        prompts, labels, options = load_benchmark(benchmark_name)

        # predict the output
        predictions = self.predict(prompts, options)

        # calculate the scores
        score_dict = self._calculate_metrics(predictions, labels)

        return score_dict

    def evaluate(self, benchmark_names):
        """Given a list of benchmark names, load and evaluate them"""
        results = {}
        for benchmark_name in benchmark_names:
            score_dict = self.evaluate_benchmark(benchmark_name=benchmark_name)
            results[benchmark_name] = score_dict

        self._pretty_print_results(results)

    def _pretty_print_results(self, results):
        """Pretty print the results"""
        for benchmark_name, score_dict in results.items():
            print(f"{benchmark_name}:")
            for metric_name, score in score_dict.items():
                print(f"\t{metric_name}: {score}")
