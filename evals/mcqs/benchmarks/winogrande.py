"""Winogrande benchmark"""

from datasets import load_dataset

SPLIT_REMAP = {"test": "validation", "validation": "train"}

import random

def load_winogrande(split="test"):
    """Load and process the benchmark"""
    base_dataset = load_dataset("allenai/winogrande", "winogrande_xs", trust_remote_code=True)[SPLIT_REMAP[split]]
    index = list(range(len(base_dataset)))
    random.shuffle(index)
    for i in index:
        sample = base_dataset[i]
        yield (
            sample["sentence"],
            sample["answer"],
            [sample["option1"] if sample["option1"] == sample["answer"] else sample["option2"]],            
        )
