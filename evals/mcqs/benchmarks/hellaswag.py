"""Hella Swag Benchmark Code: https://arxiv.org/pdf/1905.07830.pdf"""

from datasets import load_dataset
import random

def load_hellaswag(split="test"):
    """Load and process the benchmark"""
    split_map = {
        "test": "validation",
        "validation": "train",
    }
    split = split_map[split]
    base_dataset = load_dataset("Rowan/hellaswag")[split]
    index = list(range(len(base_dataset)))
    random.shuffle(index)
    for i in index:
        sample = base_dataset[i]
        ground_truth = sample["endings"][int(sample["label"])]
        yield (
            sample["ctx"],
            ground_truth,
            [ending for ending in sample["endings"] if ending != ground_truth],
        )
