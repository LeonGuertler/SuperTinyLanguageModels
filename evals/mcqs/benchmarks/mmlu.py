"""MMLU Benchmark: https://arxiv.org/abs/2009.03300"""

from datasets import load_dataset

import random

def load_mmlu(split="test"):
    """Load and process the benchmark
    
    Returns a geneator of:
    (prompt, ground_truth, fake_options)"""
    base_dataset = load_dataset("cais/mmlu","all")
    index = list(range(len(base_dataset[split])))
    random.shuffle(index)

    for i in index:
        sample = base_dataset[split][i]
        ground_truth = sample["choices"][sample["answer"]]
        fake_options = [choice for choice in sample["choices"] if choice != ground_truth]
        yield (
            sample["question"],
            ground_truth,
            fake_options,
        )
