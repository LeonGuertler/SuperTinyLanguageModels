"""Hella Swag Benchmark Code: https://arxiv.org/pdf/1905.07830.pdf"""

from datasets import load_dataset



def load_hellaswag(cache_dir="data/eval/hellaswag", split="test"):
    """Load and process the benchmark"""
    split_map = {
        "test": "validation",
        "validation": "train",
    }
    split = split_map[split]
    base_dataset = load_dataset("Rowan/hellaswag", cache_dir=cache_dir)[split]
    for sample in base_dataset:
        ground_truth = sample["endings"][int(sample["label"])]
        yield (
            sample["ctx"],
            ground_truth,
            [ending for ending in sample["endings"] if ending != ground_truth],
        )
