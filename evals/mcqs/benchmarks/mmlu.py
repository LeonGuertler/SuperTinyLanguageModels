"""MMLU Benchmark: https://arxiv.org/abs/2009.03300"""

from datasets import load_dataset

    

def load_mmlu(cache_dir="data/eval/mmlu", split="test"):
    """Load and process the benchmark
    
    Returns a geneator of:
    (prompt, ground_truth, fake_options)"""
    base_dataset = load_dataset("cais/mmlu","all", cache_dir=cache_dir)
    for sample in base_dataset[split]:
        ground_truth = sample["choices"][sample["answer"]]
        fake_options = [choice for choice in sample["choices"] if choice != ground_truth]
        yield (
            sample["question"],
            ground_truth,
            fake_options,
        )
