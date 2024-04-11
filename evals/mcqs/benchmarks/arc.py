"""ARC Benchmark: https://arxiv.org/abs/1803.05457"""

from datasets import load_dataset


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


def create_prompt(question, text_options, options):
    """
    Given the text question, text options, and the correct answer, create a prompt
    """
    option_text = []
    for i, option in enumerate(text_options):
        option_text.append(f"{options[i]}: \"{option}\"")
    return ARC_PROMPT.format(
        question=question,
        options="\n".join(option_text)
    )


def load_arc(cache_dir="data/eval/arc"):
    """ Load and process the benchmark """
    base_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir=cache_dir)["test"]
    prompts = []
    labels = []
    options = [] 
    for sample in base_dataset:
        prompts.append(
            create_prompt(
                question=sample["question"],
                text_options=sample["choices"]["text"],
                options=sample["choices"]
            )
        )
        options.append(sample["choices"])
        labels.append(str(sample["answerKey"]))

    return prompts, labels, options

