"""Winograd benchmark"""

from datasets import load_dataset

# pylint: disable=line-too-long
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
# pylint: enable=line-too-long

REMAP = {
    "0": "A",
    "1": "B",
}


def load_winograd(cache_dir="data/eval/winograd"):
    """Load and process the benchmark"""
    base_dataset = load_dataset("winograd_wsc", "wsc285", cache_dir=cache_dir)["test"]
    prompts = []
    labels = []
    for sample in base_dataset:
        prompts.append(
            WINOGRAD_PROMPT.format(
                statement=sample["text"],
                pronoun=sample["pronoun"],
                sentence=sample["quote"],
                option1=sample["options"][0],
                option2=sample["options"][1],
            )
        )
        labels.append(str(sample["label"]))
    return prompts, labels
