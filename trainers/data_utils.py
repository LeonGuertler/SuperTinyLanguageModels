"""Utilities for data"""

import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
from datasets import Features, Value


DATASET_DICT = {
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "babylm_100m": lambda: load_dataset("Sree1994/babylm_100M"), 
    "tinystories": lambda: load_dataset("roneneldan/TinyStories"),
    "openwebtext": lambda: load_dataset("Skylion007/openwebtext", trust_remote_code=True),
    "pints": lambda: load_dataset("pints-ai/Expository-Prose-V1"),
    "the_pile": lambda: load_dataset("The Pile", "pile-cc"),
    "tiny_textbooks": lambda: load_dataset("nampdn-ai/tiny-textbooks"),
    "tiny_webtext": lambda: load_dataset("nampdn-ai/tiny-webtext"),
    "tiny_bridgedict": lambda: load_dataset("nampdn-ai/tiny-bridgedict"),
    "mini_fineweb": lambda: load_dataset("nampdn-ai/mini-fineweb"),
    "openhermes-2.5": lambda: load_general_dataset(
        dataset_name="teknium/OpenHermes-2.5",
        lambda_fn=lambda x: {"text": f"Question: {x['conversations'][0]['value']}\nAnswers: {x['conversations'][1]['value']}"},
    ),
    "github-code": lambda: load_general_dataset(
        dataset_name="codeparrot/github-code",
        lambda_fn=lambda x: {"text": x["code"]}
    ),
    "competition_math": lambda: load_general_dataset(
        dataset_name="hendrycks/competition_math",
        lambda_fn=lambda x: {"text": f"Problem: {x['problem']}\nSolution: {x['solution']}"}
    ),
    "super_natural_instructions": lambda: load_general_dataset(
        dataset_name="andersonbcdefg/supernatural-instructions-2m",
        lambda_fn=lambda x: {"text": f"Question: {x['prompt']}\nAnswer: {x['response']}"}
    ),
    "tiny_codes": lambda: load_general_dataset(
        dataset_name="nampdn-ai/tiny-codes",
        lambda_fn=lambda x: {"text": f"Question: {x['prompt']}\nAnswer: {x['response']}"}
    ),
    "tiny_orca_textbooks": lambda: load_general_dataset(
        dataset_name="nampdn-ai/tiny-orca-textbooks",
        lambda_fn=lambda x: {"text": f"{x['textbook']}\n Question: {x['question']}\nAnswer: {x['response']}"}
    ),
    "tiny_lessons": lambda: load_general_dataset(
        dataset_name="nampdn-ai/tiny-lessons",
        lambda_fn=lambda x: {"text": x['textbook']}
    ),
    "mini_cot": lambda: load_general_dataset(
        dataset_name="nampdn-ai/mini-CoT-Collection",
        lambda_fn=lambda x: {"text": f"Question: {x['source']}\nAnswer: {x['rationale']} - {x['target']}"}
    ),
    "mini_ultrachat": lambda: load_general_dataset(
        dataset_name="nampdn-ai/mini-ultrachat",
        lambda_fn=lambda x: {
            "text": "".join(
                [
                    f"Question: {t}" 
                    if i % 2 == 0 else f"Answer: {t}"
                    for i, t in enumerate(x['data']) 
                ]
            )
        }
    ),
    "textbooks_are_all_you_need_lite": lambda: load_general_dataset(
        dataset_name="SciPhi/textbooks-are-all-you-need-lite",
        lambda_fn=lambda x: {"text": x["completion"]}
    ),
    "openphi_textbooks": lambda: load_general_dataset(
        dataset_name="open-phi/textbooks",
        lambda_fn=lambda x: {"text": x["markdown"]}
    ),
    "openphi_programming_books": lambda: load_general_dataset(
        dataset_name="open-phi/programming_books_llama",
        lambda_fn=lambda x: {"text": x["markdown"]}
    ),
    "natural_instructions": lambda: load_general_dataset(
        dataset_name="Muennighoff/natural-instructions",
        lambda_fn=lambda x: {"text": f"Task: Definition: {x['definition']}\nQuestion: {x['inputs']}\nAnswer: {x['targets']}"}
    ),
    "fineweb_edu_100B": lambda: load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT"),
<<<<<<< HEAD
    "fineweb_edu_10B": lambda: load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")

=======
    "fineweb_edu_10B": lambda: load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT"),
>>>>>>> f521713 (temp)


}


def load_general_dataset(dataset_name, lambda_fn):
    """
    load and re-format a huggingface dataset
    """
    dataset = load_dataset(dataset_name)
    dataset = dataset.map(lambda_fn)
    return dataset

def get_dataset_byte_size(dataset):
    """
    Get the byte size of a dataset
    """
    return sum([len(item["text"]) for item in dataset])


def load_data(dataset_name, shuffle=True):
    """Load the data"""
    assert dataset_name in DATASET_DICT, f"Dataset {dataset_name} not found!"
    dataset = DATASET_DICT[dataset_name]()

    # create dataset split
    split_dataset = dataset["train"].train_test_split(
        test_size=0.01, seed=489, shuffle=shuffle
    )

    # rename test split to val
    split_dataset["val"] = split_dataset.pop("test")

    # return the training and validation datasets
    return split_dataset