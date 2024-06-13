"""The tinystories training progression dataset"""

import random
from evals import evaluator_interface
from evals import eval_wrapper
from datasets import load_dataset
from evals.metrics import MCQ_METRIC_DICT
import torch

TASKS = [
    "byte_shuffle",
    "word_shuffle",
    "ngram_shuffle_3",
    "ngram_shuffle_5",
    "ngram_shuffle_7",
    "ngram_shuffle_12",
    "random_byte_deletion",
    "random_word_deletion",
    "random_byte_insertion",
    "random_word_insertion",
    "random_byte_substitution",
    "random_word_substitution",
    "parse_tree_shuffle"
]


def option_prompt(choice, choices):
    prompt = f"Options: {';'.join(choices)}\n Answer: {choice}"
    return prompt


def load_stories_progression(split="test"):
    """Load and process the benchmark

    Returns a geneator of:
    (prompt, ground_truth, fake_options)"""
    base_dataset = load_dataset("LeonGuertler/TinyStories_stlm_training_progress")
    base_dataset = base_dataset["train"]
    index = list(range(len(base_dataset)))
    assert split in ["validation", "test"]
    if split == "validation":
        index = index[: int(0.5 * len(index))]
    elif split == "test":
        index = index[int(0.5 * len(index)) :]
    random.shuffle(index)
    for i in index:
        sample = base_dataset[i]
        prompt = sample["first_n_words"]
        actual_continuation = sample["actual_continuation"]
        options = {
            task: sample[task] for task in TASKS
        }
        yield prompt, actual_continuation, options

class ProgressionEvaluator(evaluator_interface.EvaluationInterface):
    """Evaluator for the stories progression set"""

    def __init__(self, model):
        super().__init__(model)
        self.model_wrapper = eval_wrapper.EvalWrapper(model)

    def evaluate(self, split="test"):
        """Evaluate the model on the stories progression dataset"""
        dataset = load_stories_progression(split)
        likelihoods = []
        results = {}
        for prompt, ground_truth, options in dataset:
            all_options = list(options.items())
            options = [ground_truth] + [option for _, option in all_options] # N_options+1
            prompts = [prompt for _ in options]
            likelihoods.append(self.model_wrapper.loglikelihood(prompts, options))
        likelihoods = torch.tensor(likelihoods) # shape B * T
        gt = likelihoods[:,0] # shape B
        for i, (task) in enumerate(TASKS):
            task_results = {}
            for metric_name, metric_func in MCQ_METRIC_DICT.items():
                metric_input = torch.stack([gt, likelihoods[:,i + 1]], dim=-1)
                # shape B x 2
                task_results[metric_name]=(metric_func(metric_input))
            results[task] = task_results 
        return results

