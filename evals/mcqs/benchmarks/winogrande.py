"""Winogrande benchmark"""

from datasets import load_dataset

SPLIT_REMAP = {"test": "validation", "validation": "train", "train": "train"}
INDEX_MAP = {"1": 0, "2": 1, "3": 2, "4": 3, "A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
import random


def load_winogrande(split="test"):
    """Load and process the benchmark"""
    base_dataset = load_dataset(
        "allenai/winogrande", "winogrande_xs", trust_remote_code=True
    )[SPLIT_REMAP[split]]
    index = list(range(len(base_dataset)))
    if split == "train":
        index = index[: len(index) // 2]
    elif split == "validation":
        index = index[len(index) // 2 :]
    random.shuffle(index)
    for i in index:
        sample = base_dataset[i]
        sentence = sample["sentence"]
        options = [sample["option1"], sample["option2"]]
        ground_truth = options[INDEX_MAP[sample["answer"]]]
        options.remove(ground_truth)
        ground_truth = sentence.replace("_", ground_truth)
        options = [sentence.replace("_", option) for option in options]
        yield (
            "",
            ground_truth,
            options,
        )
