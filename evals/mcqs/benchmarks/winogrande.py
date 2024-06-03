"""Winogrande benchmark"""

from datasets import load_dataset

SPLIT_REMAP = {"test": "validation", "validation": "train"}

def load_winogrande(cache_dir="data/eval/winogrande", split="test"):
    """Load and process the benchmark"""
    base_dataset = load_dataset("allenai/winogrande", "winogrande_xs",cache_dir=cache_dir, trust_remote_code=True)[SPLIT_REMAP[split]]
    for sample in base_dataset:
        yield (
            sample["sentence"],
            sample["answer"],
            [sample["option1"] if sample["option1"] == sample["answer"] else sample["option2"]],            
        )
