"""Hella Swag Benchmark Code: https://arxiv.org/pdf/1905.07830.pdf"""

import tqdm

from datasets import load_dataset

from evals import benchmark

HELLA_SWAG_PROMPT = """Your task is to pick the most plausible continuation of a story
Example:
Story: John went to the store. He bought some milk.
Options:
A: He went home.
B: He went to the park.
C: He went to the moon.
D: He went to work.
Answer: A

Story: {story}
Options:
A: {option1}
B: {option2}
C: {option3}
D: {option4}
Answer: """

REMAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


class HellaSwag(benchmark.Benchmark):
    """HellaSwag benchmark"""

    def __init__(self, name, model, cache_dir="data/eval/hellaswag"):
        super().__init__(name, model)
        self.base_dataset = load_dataset("Rowan/hellaswag", cache_dir=cache_dir)[
            "validation"
        ]
        # preprocess the dataset

    def execute(self, batch_size=8):
        acc_metric = benchmark.AccuracyMetric()
        f1_metric = benchmark.F1Metric()
        # batch together samples for inference
        batch_prompts = []
        batch_labels = []
        for sample in tqdm.tqdm(self.base_dataset):
            options = sample["endings"]
            prompt = HELLA_SWAG_PROMPT.format(
                story=sample["ctx"],
                option1=options[0],
                option2=options[1],
                option3=options[2],
                option4=options[3],
            )
            label = REMAP[int(sample["label"])]
            batch_prompts.append(prompt)
            batch_labels.append(label)
            if len(batch_prompts) == batch_size:
                predictions = self.model.predict(batch_prompts, output_token=["A", "B"])
                targets = batch_labels
                acc_metric.batched_accumulate(predictions, targets)
                f1_metric.batched_accumulate(predictions, targets)
                batch_prompts = []
                batch_labels = []
        return acc_metric.aggregate(), f1_metric.aggregate()


if __name__ == "__main__":
    hellaswag = HellaSwag("hellaswag", benchmark.FauxModel())
    print(hellaswag.execute())
