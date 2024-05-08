"""ARC Benchmark: https://arxiv.org/abs/1803.05457"""

from datasets import load_dataset


def load_arc(cache_dir="data/eval/arc"):
    """ Load and process the benchmark """
    base_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir=cache_dir)["test"]
    prompts = []
    labels = []
    options = [] 
    for sample in base_dataset:
        prompts.append(sample["question"])
        options.append(sample["choices"])
        labels.append(str(sample["answerKey"]))

    return prompts, labels, options

