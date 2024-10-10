"""
Load a bechmark loader, given the benchmark name.
"""
import numpy as np 
from datasets import load_dataset

def get_idx_list(dataset_length, num_samples):
    """
    Given the dataset length and the number of samples,
    return a list of indices to sample from the dataset
    """
    # re-set seed every time for consistency
    np.random.seed(42)
    return np.random.choice(
        dataset_length,
        dataset_length if num_samples is None else min(num_samples, dataset_length),
        replace=False,
    ).tolist()

def load_arc_easy(version, num_samples=None):
    """ 
    Load ARC easy eval set 
    (https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy)
    """
    if version == "original":
        dataset = load_dataset("ai2_arc", "ARC-Easy", trust_remote_code=True)["test"]
    elif version == "stlm_eval":
        raise NotImplementedError("STLM eval version not implemented yet")
    
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        correct_idx = sample["choices"]["label"].index(sample["answerKey"])
        """correct_idx = next(
            (i for i, choice in enumerate(sample["choices"]["text"]) if choice == sample["answerKey"]),
            None
        )"""
        yield (
            sample["question"],
            sample["choices"]["text"][correct_idx],
            [
                sample["choices"]["text"][i]
                for i in range(len(sample["choices"]["text"]))
                if i != correct_idx
            ],
        )

def load_blimp(num_samples=None):
    """
    Load BLIMP eval set
    https://huggingface.co/datasets/WillHeld/blimp
    """
    dataset = load_dataset("WillHeld/blimp", trust_remote_code=True)["train"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        yield (
            "",
            sample["sentence_good"],
            [sample["sentence_bad"]],
        )


def load_hellaswag(version, num_samples=None):
    """
    Load hellaswag eval set
    https://huggingface.co/datasets/Rowan/hellaswag
    """
    if version == "original":
        dataset = load_dataset("Rowan/hellaswag", trust_remote_code=True)["validation"] # standard to use val
    elif version == "stlm_eval":
        raise NotImplementedError("STLM eval version not implemented yet")

    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        ground_truth_idx = int(sample["label"])
        yield (
            sample["ctx"],
            sample["endings"][ground_truth_idx],
            [ending for i,ending in enumerate(sample["endings"]) if i != ground_truth_idx],
        )


def load_mmlu(num_samples=None):
    """
    Load MMLU eval set
    https://huggingface.co/datasets/cais/mmlu
    """
    dataset = load_dataset("cais/mmlu", "all", trust_remote_code=True)["test"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        correct_idx = sample["answer"]
        yield (
            sample["question"],
            f" Answer: {sample['choices'][correct_idx]}",
            [f" Answer: {choice}" for i, choice in enumerate(sample["choices"]) if i != correct_idx],
        )

def load_winogrande(version, num_samples=None):
    """
    Load Winogrande eval set
    https://huggingface.co/datasets/allenai/winogrande
    """
    if version == "original":
        dataset = load_dataset(
            "allenai/winogrande", 
            "winogrande_xs",
            trust_remote_code=True
        )["validation"]
    elif version == "stlm_eval":
        raise NotImplementedError("STLM eval version not implemented yet")
    
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        correct_opt_num = sample["answer"]
        yield (
            "",
            sample["sentence"].replace("_", sample[f"option{correct_opt_num}"]),
            [sample["sentence"].replace("_", sample[f"option{1 if correct_opt_num == 2 else 2}"])]
        )

def load_truthful_qa_m2(version, num_samples=None):
    """
    Load the truthful QA eval set
    https://huggingface.co/datasets/TruthfulQA
    """
    if version == "original":
        dataset = load_dataset("truthful_qa", "multiple_choice", trust_remote_code=True)["validation"]
    elif version == "stlm_eval":
        raise NotImplementedError("STLM eval version not implemented yet")

    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        yield (
            sample["question"],
            sample["correct_answer"],
            sample["incorrect_answers"]
        )

def load_piqa(num_samples=None):
    """
    Load the PIQA eval set
    https://arxiv.org/abs/1911.11641
    https://huggingface.co/datasets/ybisk/piqa
    """
    dataset = load_dataset("ybisk/piqa", trust_remote_code=True)["validation"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        yield (
            sample["goal"],
            sample[f"sol{sample['label']+1}"],
            [sample[f"sol{1 if sample['label'] == 2 else 2}"]]
        )

def load_boolq(num_samples=None):
    """
    Load the BoolQ eval set
    https://arxiv.org/abs/1905.10044
    https://huggingface.co/datasets/google/boolq
    """
    dataset = load_dataset("google/boolq", trust_remote_code=True)["validation"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        yield (
            sample["question"],
            "yes" if sample["answer"] else "no",
            ["no"] if sample["answer"] else ["yes"]
        )

def load_race(version, num_samples=None):
    """
    Load the RACE eval set 
    https://aclanthology.org/D17-1082/
    https://huggingface.co/datasets/ehovy/race
    """
    ANS_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
    dataset = load_dataset(
        "ehovy/race",
        version, # middle or high school
        trust_remote_code=True
    )["validation"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        correct_idx = ANS_TO_IDX[sample["answer"]]
        yield (
            sample["article"] + f"Question: {sample['question']}",
            f"Answer: {sample['options'][correct_idx]}",
            #sample["options"][correct_idx],
            [option for i, option in enumerate(sample["options"]) if i != correct_idx]
        )

def load_openbook_qa(version, num_samples=None):
    """
    Load the OpenbookQA eval set
    https://huggingface.co/datasets/allenai/openbookqa
    """
    ANS_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
    dataset = load_dataset("allenai/openbookqa", "additional", trust_remote_code=True)["validation"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        correct_idx = ANS_TO_IDX[sample["answerKey"]]
        yield (
            sample["question_stem"] if version=="closed" else f"{sample['fact1']} {sample['question_stem']}",
            sample["choices"]["text"][correct_idx],
            [choice for i, choice in enumerate(sample["choices"]["text"]) if i != correct_idx]
        )

def load_copa(num_samples=None):
    """
    Load the Copa eval set (balanced)
    https://aclanthology.org/S12-1052/
    https://huggingface.co/datasets/pkavumba/balanced-copa
    """
    dataset = load_dataset("pkavumba/balanced-copa", trust_remote_code=True)["train"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        correct_idx = sample["label"]
        incorrect_idx = 1 if correct_idx == 0 else 0
        yield (
            sample["premise"],
            sample[f"choice{correct_idx+1}"],
            [sample[f"choice{incorrect_idx+1}"]],
        )


def load_commonsense_qa(num_samples=None):
    """
    Load the Commonsense QA eval set
    https://aclanthology.org/N19-1421/
    https://huggingface.co/datasets/tau/commonsense_qa
    """
    ANS_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}

    dataset = load_dataset("tau/commonsense_qa", trust_remote_code=True)["validation"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i]
        correct_idx = ANS_TO_IDX[sample["answerKey"]]
        yield (
            sample["question"],
            f"Answer: {sample['choices']['text'][correct_idx]}",
            [f"Answer: {choice}" for i, choice in enumerate(sample["choices"]["text"]) if i != correct_idx]
        )

def load_ewok(num_samples=None):
    """
    Load the Ewok eval set
    https://arxiv.org/abs/2405.09605v1
    https://huggingface.co/datasets/ewok-core
    """
    dataset = load_dataset("ewok-core/ewok-core-1.0", trust_remote_code=True)["test"]
    index_list = get_idx_list(len(dataset), num_samples)
    for i in index_list:
        sample = dataset[i] 
        yield(
            sample["Context1"],
            sample["Target1"],
            [sample["Target2"]]
        )




EVALS_DICT = {
    "arc_easy": lambda num_samples: load_arc_easy(
        version="original",
        num_samples=num_samples
    ),
    "stlm_eval_arc_easy": lambda num_samples: load_arc_easy(
        version="stlm_eval",
        num_samples=num_samples
    ),
    "hellaswag": lambda num_samples: load_hellaswag(
        version="original",
        num_samples=num_samples
    ),
    "stlm_eval_hellaswag": lambda num_samples: load_hellaswag(
        version="stlm_eval",
        num_samples=num_samples
    ),
    "winogrande": lambda num_samples: load_winogrande(
        version="original",
        num_samples=num_samples
    ),
    "stlm_eval_winogrande": lambda num_samples: load_winogrande(
        version="stlm_eval",
        num_samples=num_samples
    ),
    "truthful_qa": lambda num_samples: load_truthful_qa_m2(
        version="original",
        num_samples=num_samples
    ),
    "stlm_eval_truthful_qa": lambda num_samples: load_truthful_qa_m2(
        version="stlm_eval",
        num_samples=num_samples
    ),
    "blimp": load_blimp,
    "mmlu": load_mmlu,
    "piqa": load_piqa,
    "boolq": load_boolq,
    "race_middle": lambda num_samples: load_race(
        version="middle",
        num_samples=num_samples
    ),
    "race_high": lambda num_samples: load_race(
        version="high",
        num_samples=num_samples
    ),
    "openbook_qa_open": lambda num_samples: load_openbook_qa(
        version="open",
        num_samples=num_samples
    ),
    "openbook_qa_closed": lambda num_samples: load_openbook_qa(
        version="closed",
        num_samples=num_samples
    ),
    "copa": lambda num_samples: load_copa(
        num_samples=num_samples
    ),
    "commonsense_qa": lambda num_samples: load_commonsense_qa(
        num_samples=num_samples
    ),
    "ewok": lambda num_samples: load_ewok(
        num_samples=num_samples
    )
}




def load_benchmark(benchmark_name, num_samples):
    """
    Given the benchmark name, build the benchmark
    """
    assert benchmark_name in EVALS_DICT, \
        f"Benchmark {benchmark_name} not found. The available benchmarks are: {list(EVALS_DICT.keys())}"
    return EVALS_DICT[benchmark_name](
        num_samples=num_samples
    )
