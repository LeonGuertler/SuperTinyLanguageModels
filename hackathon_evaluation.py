"""
Evaluation Code.
"""

import torch 
import sympy
from hackathon_utils import compare_answers
from datasets import load_dataset
from models.build_models import build_model


def load_math():
    """
    Load the MATH eval set
    https://huggingface.co/datasets/lighteval/MATH

    Args:
        num_samples (Optional[int]): Number of samples to load. If None, load all.
        seed (Optional[int]): Seed for random sampling.

    Yields:
        Tuple[str, str, List[str]]: (question, answer)
    """
    dataset = load_dataset("lighteval/MATH", "all", trust_remote_code=True)["test"]
    for i in range(len(dataset)):
        yield(
            sample[i]["problem"],
            sample[i]["solution"]
        )
    

# load the model
model, _ = build_model(model_cfg={
    "model_string": "openai-community/gpt2",
    "core_model_type": "hf_core",
    "embedding_model_type": "hf_embedder",
    "tokenizer_name": "hf_tokenizer",
    "model_shell_type": "standard",
    "lm_head_type": "hf_head"
})


def generate_reply(problem):
    return problem




total, correct = 0, 0
for i, (problem, solution) in enumerate(load_math()):
    # get model answer
    model_answer = generate_reply(problem)

    # evaluate model answer
    correct += compare_answers(
        y_true=solution,
        y_pred=model_answer
    )
    total += 1



print(f"Accuracy: {correct/total}")