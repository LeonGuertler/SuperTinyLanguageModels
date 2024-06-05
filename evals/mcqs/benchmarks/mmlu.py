"""MMLU Benchmark: https://arxiv.org/abs/2009.03300"""

import random

from datasets import load_dataset


def option_prompt(choice, choices):
    prompt = f"Options: {';'.join(choices)}\n Answer: {choice}"
    return prompt


def load_mmlu(split="test"):
    """Load and process the benchmark

    Returns a geneator of:
    (prompt, ground_truth, fake_options)"""
    base_dataset = load_dataset("cais/mmlu", "all")
    index = list(range(len(base_dataset[split])))
    random.shuffle(index)

    for i in index:
        sample = base_dataset[split][i]
        ground_truth = sample["choices"][sample["answer"]]
        ground_truth = option_prompt(ground_truth, sample["choices"])
        fake_options = [
            choice for choice in sample["choices"] if choice != ground_truth
        ]
        fake_options = [
            option_prompt(choice, sample["choices"]) for choice in fake_options
        ]
        yield (
            sample["question"],
            ground_truth,
            fake_options,
        )
