"""MMLU benchmark: https://arxiv.org/pdf/2009.03300.pdf"""

import tqdm
import os
import requests
import glob
import pandas as pd

from evals import benchmark

MMLU_PROMPT = """This problem is from the MMLU benchmark. In particular from the {title} subset.

Here is an example of a problem from this set:
Question: "{example_problem}"
Choices:
A: {example_option1}
B: {example_option2}
C: {example_option3}
D: {example_option4}
Answer: {example_answer}

Here is the problem you need to solve:
Question: "{question}"
Choices:
A: {option1}
B: {option2}
C: {option3}
D: {option4}
Answer: """

MMLU_RAW_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


class MMLUBenchmark(benchmark.Benchmark):
    """MMLU benchmark"""

    def __init__(self, name, model, cache_dir="data/eval/mmlu"):
        super().__init__(name, model)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            try:
                r = requests.get(MMLU_RAW_URL, allow_redirects=True)
                open(f"{cache_dir}/data.tar", "wb").write(r.content)
                os.system(f"tar -xvf {cache_dir}/data.tar -C {cache_dir}")
            except Exception as e:
                print(f"Failed to download and extract MMLU data: {e}")

    def execute(self):
        # search test data for all files
        results = {}
        for filename in glob.glob("data/eval/mmlu/test/*.csv"):
            train_filename = filename.replace("test", "dev")
            title = filename.split("/")[-1].split(".")[0]
            results[title] = self.execute_subset(filename, title, train_filename)

        return results

    def execute_subset(self, filename, title, train_filename):
        """Run the benchmark."""
        acc_metric = benchmark.AccuracyMetric()
        # f1_metric = benchmark.F1Metric()
        # grab example from train
        example = pd.read_csv(train_filename, header=None, quotechar='"').values[0]
        example_problem = example[0]
        example_option1 = example[1]
        example_option2 = example[2]
        example_option3 = example[3]
        example_option4 = example[4]
        example_answer = example[5]
        test_data = pd.read_csv(filename, header=None, quotechar='"')
        prompts = []
        labels = []
        for problem in tqdm.tqdm(test_data.values):
            question = problem[0]
            option1 = problem[1]
            option2 = problem[2]
            option3 = problem[3]
            option4 = problem[4]
            answer = problem[5]
            prompt = MMLU_PROMPT.format(
                title=title,
                example_problem=example_problem,
                example_option1=example_option1,
                example_option2=example_option2,
                example_option3=example_option3,
                example_option4=example_option4,
                example_answer=example_answer,
                question=question,
                option1=option1,
                option2=option2,
                option3=option3,
                option4=option4,
            )
            label = answer
            prompts.append(prompt)
            labels.append(label)
            if len(prompts) == 8:
                predictions = self.model.predict(
                    prompts, options=["A", "B", "C", "D"]
                )
                targets = labels
                acc_metric.batched_accumulate(predictions, targets)
                # f1_metric.batched_accumulate(predictions, targets)
                prompts = []
                labels = []
            # f1_metric.accumulate(predictions, targets)
        return acc_metric.aggregate()


if __name__ == "__main__":
    mmlu = MMLUBenchmark("mmlu", benchmark.FauxModel())
    res = mmlu.execute()
    for k, v in res.items():
        print(f"{k}: {v}")
    avg = sum(res.values()) / len(res)
    print(f"Average: {avg}")
