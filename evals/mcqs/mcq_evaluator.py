"""
An evaluator that, for a given dataset and model calculates the path 
probabilities of each of the answers and returns a score.
"""
import torch 

from evals.mcqs.load_benchmarks import load_benchmark
from evals.metrics import MCQ_METRIC_DICT 


class MCQEvaluator:
    """
    Base MCQ evaluator class that evalutes
    models based on the path probabilities given
    for each answer.
    """
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

        # extract tokenizer from model 
        self.tokenizer = model.tokenizer

        # make sure the model is in eval mode 
        self.model.eval()

    @torch.no_grad()
    def _path_probability(self, prompt, options):
        """
        Given a prompt and a list of options, calculate the path probabilities
        for each of the options.
        (average per token log probability)
        """
        probabilities = {} 

        for marker, option in options:
            # combine the prompt with each option 
            full_input = f"{prompt} {option}"

            # tokenize
            input_ids = self.tokenizer(full_input, return_tensors="pt")

            # pass through model 
            logits = self.model(input_ids)

            # shift to align with the target
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = input_ids[:, 1:].contiguous()

            # Calculate log probabilities using negative log likelihood
            log_probs = torch.nn.functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                reduction='none'
            )

            # Reshape to the original token dimensions and sum the log probs for each sequence
            log_probs_per_token = log_probs.view(input_ids.size(0), -1)
            total_log_probs = log_probs_per_token.sum(dim=1)
            
            # Compute the average log probability per token
            avg_log_prob_per_token = total_log_probs / (input_ids.size(1) - 1)
            
            # Convert to the actual probability scale by exponentiating
            avg_probability = torch.exp(-avg_log_prob_per_token).item()
            
            probabilities[marker] = avg_probability

        return probabilities
    
    def _get_path_probabilities(self, prompts, labels, options):
        """
        Calculate the path probabilities for each of the options
        for each prompt.
        """
        predictions = []

        for prompt, label, option in zip(prompts, labels, options):
            # calculate the path probabilities
            path_probabilities = self._path_probability(prompt, option)

            # append the path probabilities to the predictions
            predictions.append(path_probabilities)

        return predictions

    def _calculate_metrics(self, predictions, targets):
        """
        Calculate the metrics for the model
        """
        score_dict = {}

        for metric_name in MCQ_METRIC_DICT.keys():
            score_dict[metric_name] = MCQ_METRIC_DICT[metric_name](
                predictions=predictions,
                targets=targets
            )
        
        return score_dict
    
    def evaluate_benchmark(self, benchmark_name):
        """ Evaluate model performance on a specific benchmark"""
        # load the benchmark_loader
        prompts, labels, options = load_benchmark(benchmark_name)

        # predict the output
        predictions = self._get_path_probabilities(prompts, labels, options)

        # calculate the scores
        score_dict = self._calculate_metrics(predictions, labels)

        return score_dict 
    
    def evaluate(self, benchmark_names):
        """ Given a list of benchmark names, load and evaluate them """
        results = {}
        for benchmark_name in benchmark_names:
            score_dict = self.evaluate_benchmark(
                benchmark_name=benchmark_name
            )
            results[benchmark_name] = score_dict

        self._pretty_print_results(results)

    def _pretty_print_results(self, results):
        """ Pretty print the results """
        for benchmark_name, score_dict in results.items():
            print(f"{benchmark_name}:")
            for metric_name, score in score_dict.items():
                print(f"\t{metric_name}: {score}")

