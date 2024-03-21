"""Nonsense Benchmark: https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/nonsense_words_grammar"""

import tqdm

from datasets import load_dataset

from evals import benchmark


class Nonsense(benchmark.Benchmark):
    """Nonsense benchmark from bigbench"""

    def __init__(self, name, model, cache_dir="data/eval/nonsense"):
        super().__init__(name, model)
        self.base_dataset = load_dataset("tasksource/bigbench", "nonsense_words_grammar", cache_dir=cache_dir)["validation"]
        # preprocess the dataset

    def execute(self, batch_size=8):
        acc_metric = benchmark.AccuracyMetric()
        # batch together samples for inference
        batch_prompts = []
        batch_labels = []
        output_tokens = []
        for sample in tqdm.tqdm(self.base_dataset):
            prompt = sample["inputs"]
            label = str(sample["targets"][0])
            batch_prompts.append(prompt)
            batch_labels.append(label)
            output_tokens.append(sample["multiple_choice_targets"])
            if len(batch_prompts) == batch_size:
                predictions = self.model.predict(batch_prompts, output_token=output_tokens)
                targets = batch_labels
                acc_metric.batched_accumulate(predictions, targets)
                batch_prompts = []
                batch_labels = []
                output_tokens = []
        return acc_metric.aggregate()


if __name__ == "__main__":
    arc = Nonsense("nonsense", benchmark.FauxModel())
    print(arc.execute())
