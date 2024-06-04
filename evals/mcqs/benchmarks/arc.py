"""ARC Benchmark: https://arxiv.org/abs/1803.05457"""

import random

from datasets import load_dataset


def _split_options(options, labels, answer_key):
    """Split the options according to whether label matches answer key"""
    true_idx = ...  # assume there is only one correct option
    for idx, label in enumerate(labels):
        if label == answer_key:
            true_idx = idx
    return options[true_idx], options[:true_idx] + options[true_idx + 1 :]


def load_arc(split="test"):
    """Load and process the benchmark

    Returns a geneator of:
    (prompt, ground_truth, fake_options)"""
    base_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")[split]
    index = list(range(len(base_dataset)))
    random.shuffle(index)

    for i in index:
        sample = base_dataset[i]
        ground_truth, fake_options = _split_options(
            sample["choices"]["text"],
            sample["choices"]["label"],
            sample["answerKey"],
        )
        yield (
            sample["question"],
            ground_truth,
            fake_options,
        )
