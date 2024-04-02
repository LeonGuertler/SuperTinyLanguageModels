"""Vitamin C benchmark: https://aclanthology.org/2021.naacl-main.52/"""

import tqdm
from datasets import load_dataset

from evals import benchmark

# pylint: disable=line-too-long
VITAMINC_PROMPT = """Read this claim and evidence and decide whether the evidence supports the claim.
Your answer should be either A,B,C where:
A: Supports
B: Does not support
C: Refutes

Example:
Claim: Vitamin C is good for you. 
Evidence: Vitamin C is a vitamin and vitamins are good for you.
Answer: A

Claim: "{claim}"
Evidence: "{evidence}"
Answer: """
# pylint: enable=line-too-long

REMAP = {
    "REFUTES": "C",
    "NOT ENOUGH INFO": "B",
    "SUPPORTS": "A",
}


class VitaminC(benchmark.Benchmark):
    """Vitamin C benchmark"""

    def __init__(self, name, model, cache_dir="data/eval/vitaminc"):
        super().__init__(name, model)
        self.base_dataset = load_dataset("tals/vitaminc", cache_dir=cache_dir)["test"]
        # preprocess the dataset

    def execute(self, batch_size=8):
        acc_metric = benchmark.AccuracyMetric()
        f1_metric = benchmark.F1Metric()
        # batch together samples for inference
        batch_prompts = []
        batch_labels = []
        for sample in tqdm.tqdm(self.base_dataset):
            prompt = VITAMINC_PROMPT.format(
                claim=sample["claim"], evidence=sample["evidence"]
            )
            label = REMAP[str(sample["label"])]
            batch_prompts.append(prompt)
            batch_labels.append(label)
            if len(batch_prompts) == batch_size:
                predictions = self.model.predict(batch_prompts, options=["A", "B"])
                targets = batch_labels
                acc_metric.batched_accumulate(predictions, targets)
                f1_metric.batched_accumulate(predictions, targets)
                batch_prompts = []
                batch_labels = []
        return acc_metric.aggregate(), f1_metric.aggregate()


if __name__ == "__main__":
    vitaminc = VitaminC("vitaminc", benchmark.FauxModel())
    print(vitaminc.execute())
