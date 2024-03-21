"""ARC Benchmark: https://arxiv.org/abs/1803.05457"""

import tqdm

from datasets import load_dataset

from evals import benchmark

ARC_PROMPT = """Read this question and use your common sense to answer it.
Your answer should be either A,B,C where:
A: Supports
B: Does not support
C: Refutes

Example:
Question: 	
"George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?"
Options:
A: "dry palms"
B: "wet palms"
C: "palms covered with oil"
D: "palms covered with lotion"
Answer: A

Question: "{question}"
Options:
{options}
Answer: """


class ARC(benchmark.Benchmark):
    """ARC benchmark, we use the easy version"""

    def __init__(self, name, model, cache_dir="data/eval/arc"):
        super().__init__(name, model)
        self.base_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir=cache_dir)["test"]
        # preprocess the dataset

    def execute(self, batch_size=8):
        acc_metric = benchmark.AccuracyMetric()
        # batch together samples for inference
        batch_prompts = []
        batch_labels = []
        batch_options = []
        for sample in tqdm.tqdm(self.base_dataset):
            options = sample["choices"]["label"]
            option_text = []
            for i, option in enumerate(sample["choices"]["text"]):
                option_text.append(f"{options[i]}: \"{option}\"")
            prompt = ARC_PROMPT.format(
                question=sample["question"],
                options="\n".join(option_text)
            )
            label = str(sample["answerKey"])
            batch_prompts.append(prompt)
            batch_labels.append(label)
            batch_options.append(options)
            if len(batch_prompts) == batch_size:
                predictions = self.model.predict(batch_prompts, output_token=batch_options)
                targets = batch_labels
                acc_metric.batched_accumulate(predictions, targets)
                batch_prompts = []
                batch_labels = []
                batch_options = []
        return acc_metric.aggregate()


if __name__ == "__main__":
    arc = ARC("arc", benchmark.FauxModel())
    print(arc.execute())
