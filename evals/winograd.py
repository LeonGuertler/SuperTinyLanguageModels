""""""

import tqdm

from datasets import load_dataset

from evals import benchmark


WINOGRAD_PROMPT = """A Winograd schema is a pair of sentences that differ in only one or two words and that contain an ambiguity that is resolved in opposite ways in the two sentences and requires the use of world knowledge and reasoning for its resolution.
The schema takes its name from a well-known example by Terry Winograd:

Statement: The city councilmen refused the demonstrators a permit because they feared violence.

Who does "they" refer to in this "they feared violence"?
A: The city councilmen
B: The demonstrators
Answer: A


Statement: "{statement}"

Who does "{pronoun}" refer to in "{sentence}"?
A: {option1}
B: {option2}
Answer: """

REMAP = {
    "0": "A",
    "1": "B",
}


class Winograd(benchmark.Benchmark):
    """Winograd benchmark"""

    def __init__(self, name, model, cache_dir="data/eval/winograd"):
        super().__init__(name, model)
        self.base_dataset = load_dataset("winograd_wsc", "wsc285", cache_dir=cache_dir)[
            "test"
        ]
        # preprocess the dataset

    def execute(self, batch_size=8):
        metric = benchmark.AccuracyMetric()
        # batch together samples for inference
        batch_prompts = []
        batch_labels = []
        for sample in tqdm.tqdm(self.base_dataset):
            prompt = WINOGRAD_PROMPT.format(
                statement=sample["text"],
                pronoun=sample["pronoun"],
                sentence=sample["quote"],
                option1=sample["options"][0],
                option2=sample["options"][1],
            )
            label = REMAP[str(sample["label"])]
            batch_prompts.append(prompt)
            batch_labels.append(label)
            if len(batch_prompts) == batch_size:
                predictions = self.model.predict(batch_prompts, output_token=["A", "B"])
                targets = batch_labels
                metric.batched_accumulate(predictions, targets)
                batch_prompts = []
                batch_labels = []
        return metric.aggregate()


if __name__ == "__main__":
    winograd = Winograd("winograd", benchmark.FauxModel())
    print(winograd.execute())
