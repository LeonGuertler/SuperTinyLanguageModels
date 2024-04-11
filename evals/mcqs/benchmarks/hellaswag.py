"""Hella Swag Benchmark Code: https://arxiv.org/pdf/1905.07830.pdf"""

from datasets import load_dataset


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
{options}
Answer: """

REMAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

def create_prompt(question, text_options, options):
    """
    Given the text question, text options, and the correct answer, create a prompt
    """
    option_text = []
    for i, option in enumerate(text_options):
        option_text.append(f"{options[i]}: \"{option}\"")
    return HELLA_SWAG_PROMPT.format(
        question=question,
        options="\n".join(option_text)
    )


def load_hellaswag(cache_dir="data/eval/hellaswag"):
    """ Load and process the benchmark """
    base_dataset = load_dataset("Rowan/hellaswag", cache_dir=cache_dir)["validation"]
    prompts = []
    labels = []
    options = []
    for sample in base_dataset:
        prompts.append(
            create_prompt(
                question=sample["ctx"],
                text_options=sample["endings"],
                options=['A', 'B', 'C', 'D'][:len(sample["endings"])],
            )
        )
        options.append(['A', 'B', 'C', 'D'][:len(sample["endings"])])
        labels.append(str(sample["label"]))

    return prompts, labels, options

