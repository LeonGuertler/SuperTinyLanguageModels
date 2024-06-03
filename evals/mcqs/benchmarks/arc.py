"""ARC Benchmark: https://arxiv.org/abs/1803.05457"""

from datasets import load_dataset

def _split_options(options, labels, answer_key):
    """Split the options according to whether label matches answer key"""
    true_idx = ...# assume there is only one correct option
    for idx, label in enumerate(labels):
        if label==answer_key:
            true_idx = idx
    return options[true_idx], options[:true_idx] + options[true_idx+1:]

def load_arc(cache_dir="data/eval/arc", split="test"):
    """Load and process the benchmark
    
    Returns a geneator of:
    (prompt, ground_truth, fake_options)"""
    base_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir=cache_dir)[
        split
    ]

    for sample in base_dataset:
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
