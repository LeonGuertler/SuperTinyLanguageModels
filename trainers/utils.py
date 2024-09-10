"""Utilities for the trainer"""

import importlib
from prettytable import PrettyTable
import inspect
import os
import pkgutil

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
import pyarrow as pa
import torch.distributed as dist

def set_seed(seed):
    """Setup the trainer"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def create_folder_structure(path_config):
    """
    Create all necessary folders for training.
    """
    if not os.path.exists(path_config["data_dir"]):
        os.makedirs(path_config["data_dir"])

    if not os.path.exists(path_config["checkpoint_dir"]):
        os.makedirs(path_config["checkpoint_dir"])

def create_stlm_data_mix():
    """
    A small custom datamix for STLM models containing:
    - simple English Wikipedia
    - Python Code (Deepmind Code Contest) - sampled for easy questions
    - technical QA style (StackExchange)
    """
    # Load simple English Wikipedia
    wiki = load_dataset("wikimedia/wikipedia", "20231101.simple")["train"]

    # Add a "text" column for simple English Wikipedia
    wiki = wiki.map(lambda x: {"text": x["text"]})

    # Load Python code from DeepMind Code Contests
    code_dataset = load_dataset("jtatman/python-code-dataset-500k")["train"]
    code_dataset = code_dataset.map(lambda x: {"text": f"Instruction: {x['instruction']}\nOutput: {x['output']}"})


    # Load technical QA style data from StackExchange
    openhermes = load_dataset("teknium/OpenHermes-2.5")["train"]

    # Transform to have a "text" column with both question and answers
    openhermes = openhermes.map(lambda x: {"text": f"Question: {x['conversations'][0]['value']}\nAnswers: {x['conversations'][1]['value']}"})

    # Add tiny stories
    tiny_stories = load_dataset("roneneldan/TinyStories")["train"]


    # Calculate and print the distribution of string lengths
    def calculate_length_distribution(dataset):
        lengths = [len(item["text"]) for item in dataset]
        return sum(lengths), lengths

    wiki_length, wiki_lengths = calculate_length_distribution(wiki)
    python3_code_length, python3_code_lengths = calculate_length_distribution(code_dataset)
    openhermes_length, openhermes_lengths = calculate_length_distribution(openhermes)
    tiny_stories_length, tiny_stories_lengths = calculate_length_distribution(tiny_stories)

    total_length = wiki_length + python3_code_length + openhermes_length + tiny_stories_length

    print(f"Wiki Text Length: {wiki_length} ({wiki_length/total_length*100:.2f}%)")
    print(f"Python Code Text Length: {python3_code_length} ({python3_code_length/total_length*100:.2f}%)")
    print(f"openhermes Text Length: {openhermes_length} ({openhermes_length/total_length*100:.2f}%)")

    # Concatenate datasets
    combined_dataset = concatenate_datasets([wiki, code_dataset, openhermes, tiny_stories])

    combined_dataset = DatasetDict({
        "train": combined_dataset,
    })

    return combined_dataset


def load_github_code_dataset():
    """
    load and re-format the github code dataset
    https://huggingface.co/datasets/codeparrot/github-code
    """
    dataset = load_dataset("codeparrot/github-code") 

    # rename "code" column to "text" column
    dataset = dataset.map(lambda x: {"text": x["code"]})["train"]

    #dataset = DatasetDict({
    #    "train": dataset,
    #})


    return dataset

def load_competition_math_dataset():
    """
    load and re-format the competition math dataset
    https://huggingface.co/datasets/hendrycks/competition_math
    """
    dataset = load_dataset("hendrycks/competition_math") 

    # format the problem and solution into a single "text" column
    dataset = dataset.map(lambda x: {"text": f"Problem: {x['problem']}\nSolution: {x['solution']}"})


    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset


def load_open_hermes():
    """
    Load and format the OpenHermes dataset
    https://huggingface.co/datasets/teknium/OpenHermes-2.5
    """
    dataset = load_dataset("teknium/OpenHermes-2.5")["train"]

    # format the prompt and answer into a single "text" column 
    dataset = dataset.map(lambda x: {"text": f"Question: {x['conversations'][0]['value']}\nAnswers: {x['conversations'][1]['value']}"})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset

def load_supernatural_instructions():
    """
    Load and format the supernatural instructions dataset
    https://huggingface.co/datasets/andersonbcdefg/supernatural-instructions-2m
    """
    dataset = load_dataset("andersonbcdefg/supernatural-instructions-2m")["train"]

    # format the prompt and answer into a single "text" column 
    dataset = dataset.map(lambda x: {"text": f"Question: {x['prompt']}\nAnswer: {x['response']}"})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset

def load_mini_cot():
    """
    Load and format the mini-cot dataset
    https://huggingface.co/datasets/nampdn-ai/mini-CoT-Collection
    """
    dataset = load_dataset("nampdn-ai/mini-CoT-Collection")["train"]

    # format the prompt and answer into a single "text" column
    dataset = dataset.map(lambda x: {"text": f"Question: {x['source']}\nAnswer: {x['rationale']} - {x['target']}"})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset

def load_mini_ultrachat():
    """
    Load and format the mini-ultrachat dataset
    https://huggingface.co/datasets/nampdn-ai/mini-ultrachat
    """
    dataset = load_dataset("nampdn-ai/mini-ultrachat")["train"]

    # format the iterative prompt and answer into a single "text" column
    dataset = dataset.map(
        lambda x: {
            "text": "".join(
                [
                    f"Question: {t}" 
                    if i % 2 == 0 else f"Answer: {t}"
                    for i, t in enumerate(x['data']) 
                ]
            )
        }
    )

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset

def load_textbooks_are_all_you_need_lite():
    """
    Load and format the textbooks are all you need lite dataset
    https://huggingface.co/datasets/SciPhi/textbooks-are-all-you-need-lite
    """
    dataset = load_dataset("SciPhi/textbooks-are-all-you-need-lite")["train"]

    # format the data 
    dataset = dataset.map(lambda x: {"text": x["completion"]})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset

def load_openphi_textbooks():
    """
    Load and format the openphi textbooks dataset
    https://huggingface.co/datasets/open-phi/textbooks
    """
    dataset = load_dataset("open-phi/textbooks")["train"]

    # format the data 
    dataset = dataset.map(lambda x: {"text": x["markdown"]})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset


def load_openphi_programming_books():
    """
    Load and format the openphi programming textbooks dataset
    https://huggingface.co/datasets/open-phi/programming_books_llama
    """
    dataset = load_dataset("open-phi/programming_books_llama")["train"]

    # format the data
    dataset = dataset.map(lambda x: {"text": x["markdown"]})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset


def load_tiny_codes():
    """
    Load and format the tiny-codes dataset
    https://huggingface.co/datasets/nampdn-ai/tiny-codes
    """
    dataset = load_dataset("nampdn-ai/tiny-codes")["train"]

    # format the data
    dataset = dataset.map(lambda x: {"text": f"Question: {x['prompt']}\nAnswer: {x['response']}"})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset

def load_tiny_orca_textbooks():
    """
    Load and format the tiny-orca dataset
    https://huggingface.co/datasets/nampdn-ai/tiny-orca-textbooks
    """
    dataset = load_dataset("nampdn-ai/tiny-orca-textbooks")["train"]

    # format the data
    dataset = dataset.map(lambda x: {"text": f"{x['textbook']}\n Question: {x['question']}\nAnswer: {x['response']}"})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset

def load_tiny_lessons():
    """
    Load and format the tiny-lessons dataset
    https://huggingface.co/datasets/nampdn-ai/tiny-lessons
    """
    dataset = load_dataset("nampdn-ai/tiny-lessons")["train"]

    # format the data
    dataset = dataset.map(lambda x: {"text": x['textbook']})

    dataset = DatasetDict({
        "train": dataset,
    })

    return dataset


def get_dataset_byte_size(dataset):
    """
    Get the byte size of a dataset
    """
    return sum([len(item["text"]) for item in dataset])

def create_tiny_pile(verbose=True):
    """
    Combine multiple high-quality tiny datasets to create the tiny-pile dataset
    1. tiny_textbooks
    2. tiny_codes
    3. tiny_orca_textbooks
    4. tiny_webtext (exclude for now)
    5. tiny_lessons
    6. mini_fineweb (exclude for now)
    7. mini_cot
    8. mini_ultrachat
    9. textbooks_are_all_you_need_lite
    10. openphi_textbooks
    11. openphi_programming_books
    """
    tiny_textbooks = load_dataset("nampdn-ai/tiny-textbooks")["train"].remove_columns(["source", "s", "len", "idx", "textbook"])
    tiny_codes = load_tiny_codes()["train"].remove_columns(["prompt", "main_topic", "subtopic", "adjective", "action_verb", "scenario", "target_audience","programming_language", "common_sense_topic", "idx", "response"])
    tiny_orca_textbooks = load_tiny_orca_textbooks()["train"].remove_columns(["id", "prompt", "textbook", "question", "response"])
    tiny_lessons = load_tiny_lessons()["train"].remove_columns(["source", "s", "len", "idx", "textbook"])
    #mini_fineweb = load_dataset("nampdn-ai/mini-fineweb")["train"]
    mini_cot = load_mini_cot()["train"].remove_columns(["source", "target", "rationale", "task", "type"])
    mini_ultrachat = load_mini_ultrachat()["train"].remove_columns(["id", "data"])
    textbooks_are_all_you_need_lite = load_textbooks_are_all_you_need_lite()["train"].remove_columns(["formatted_prompt", "completion", "first_task", "second_task", "last_task", "notes", "title", "model", "temperature"])
    openphi_textbooks = load_openphi_textbooks()["train"].remove_columns(["topic", "model", "concepts", "outline", "markdown", "field", "subfield", "rag"])
    openphi_programming_books = load_openphi_programming_books()["train"].remove_columns(["topic", "outline", "concepts", "queries", "context", "markdown", "model"])


    # Ensure all datasets have the same column type
    datasets = [
        tiny_textbooks, tiny_codes, tiny_orca_textbooks, tiny_lessons,
        mini_cot, mini_ultrachat, textbooks_are_all_you_need_lite,
        openphi_textbooks, openphi_programming_books
    ]

    # Cast the "text" column to pa.large_string() for each dataset
    datasets = [dataset.cast_column("text", pa.large_string()) for dataset in datasets]

    # Now concatenate the datasets
    combined_dataset = concatenate_datasets(datasets)

    if verbose:
        """
        For each dataset, count the bytes and print the percentage contribution
        of each to the full pile dataset
        """
        dataset_sizes = {
            "tiny_textbooks": get_dataset_byte_size(tiny_textbooks),
            "tiny_codes": get_dataset_byte_size(tiny_codes),
            "tiny_orca_textbooks": get_dataset_byte_size(tiny_orca_textbooks),
            "tiny_lessons": get_dataset_byte_size(tiny_lessons),
            #"mini_fineweb": get_dataset_byte_size(mini_fineweb),
            "mini_cot": get_dataset_byte_size(mini_cot),
            "mini_ultrachat": get_dataset_byte_size(mini_ultrachat),
            "textbooks_are_all_you_need_lite": get_dataset_byte_size(textbooks_are_all_you_need_lite),
            "openphi_textbooks": get_dataset_byte_size(openphi_textbooks),
            "openphi_programming_books": get_dataset_byte_size(openphi_programming_books),
        }

        total_size = sum(dataset_sizes.values())
        table = PrettyTable(["Dataset", "Byte Size", "Percentage"])
        for dataset_name, size in dataset_sizes.items():
            table.add_row([dataset_name, size, size/total_size*100])
        print(table)


    combined_dataset = DatasetDict({
        "train": combined_dataset,
    })

    return combined_dataset



DATASET_DICT = {
    "debug": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "babylm_100m": lambda: load_dataset("Sree1994/babylm_100M"), # https://babylm.github.io/
    "tinystories": lambda: load_dataset("roneneldan/TinyStories"), # https://huggingface.co/datasets/roneneldan/TinyStories
    "stlm": create_stlm_data_mix,
    "openhermes-2.5": lambda: load_open_hermes(),
    "openwebtext": lambda: load_dataset("Skylion007/openwebtext", trust_remote_code=True),
    "github-code": lambda: load_github_code_dataset(),
    "competition_math": lambda: load_competition_math_dataset(),
    "pints": lambda: load_dataset("pints-ai/Expository-Prose-V1"),
    "super_natural_instructions": lambda: load_supernatural_instructions(),
    "the_pile": lambda: load_dataset("The Pile", "pile-cc"),
    "tiny_textbooks": lambda: load_dataset("nampdn-ai/tiny-textbooks"),
    "tiny_codes": lambda: load_tiny_codes(),
    #"tiny_math_textbooks": lambda: load_dataset("nampdn-ai/tiny-math-textbooks"),
    "tiny_orca_textbooks": lambda: load_tiny_orca_textbooks(),
    "tiny_webtext": lambda: load_dataset("nampdn-ai/tiny-webtext"),
    "tiny_lessons": lambda: load_tiny_lessons(),
    "tiny_bridgedict": lambda: load_dataset("nampdn-ai/tiny-bridgedict"),
    "mini_fineweb": lambda: load_dataset("nampdn-ai/mini-fineweb"),
    "mini_cot": lambda: load_mini_cot(),
    "mini_ultrachat": lambda: load_mini_ultrachat(),
    "textbooks_are_all_you_need_lite": lambda: load_textbooks_are_all_you_need_lite(),
    "openphi_textbooks": lambda: load_openphi_textbooks(),
    "openphi_programming_books": lambda: load_openphi_programming_books(),
    "tiny_pile": lambda: create_tiny_pile(),


}


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

    if dataset_name == "debug":
        split_dataset["train"] = split_dataset["train"].select(range(2048))

    # return the training and validation datasets
    return split_dataset


def get_classes_from_module(module_name):
    """
    Get a list of classes defined in a module or package.

    Args:
        module_name (str): The name of the module or package.

    Returns:
        list: A list of classes defined in the module or package.
    """
    module = importlib.import_module(module_name)
    classes = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if inspect.getmodule(obj) == module:
            classes.append(obj)

    return classes


def get_classes_from_package(package_name):
    """
    Get a list of classes defined in a package and its subpackages.

    Args:
        package_name (str): The name of the package.

    Returns:
        list: A list of classes defined in the package and its subpackages.
    """
    package = importlib.import_module(package_name)
    classes = get_classes_from_module(package_name)

    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        classes.extend(get_classes_from_module(module_name))

    return classes


def register_backward_hooks(tensor, module_name):
    """Registers hooks to profile the backward pass of a tensor."""
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:

        def backward_hook(grad):
            with torch.autograd.profiler.record_function(f"{module_name}.backward"):
                return grad

        tensor.register_hook(backward_hook)


def profilize(model, classes=None):
    """Recursively add hooks to the model for recording PyTorch profiler traces with module names"""
    if classes is None:
        classes = get_classes_from_package("models")
        classes += get_classes_from_package("models.components.layers")
        print(f"Found classes for profiling: {classes}")

    for module in model.children():
        if isinstance(module, torch.nn.Module):
            profilize(module, classes=classes)
        if isinstance(module, torch.nn.ModuleDict):
            for sub_module in module.values():
                profilize(sub_module, classes=classes)
        if isinstance(module, torch.nn.ModuleList):
            for sub_module in module:
                profilize(sub_module, classes=classes)

    if (
        hasattr(model, "forward")
        and any(isinstance(model, cls) for cls in classes)
        and not hasattr(model, "old_forward")
    ):
        model.old_forward = model.forward
        print(f"added forward profiling wrapper for {model.__class__.__name__}")

        def forward_wrapper(*args, **kwargs):
            nested_module_name = model.__class__.__name__
            with torch.autograd.profiler.record_function(
                f"{nested_module_name}.forward"
            ):
                outputs = model.old_forward(*args, **kwargs)
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    register_backward_hooks(output, nested_module_name)
            else:
                register_backward_hooks(outputs, nested_module_name)
            return outputs

        model.forward = forward_wrapper

def is_dist():
    """
    Check if the current process is distributed.
    """
    return dist.is_initialized()

def aggregate_value(value, device = torch.device("cuda")): 
    """
    Since using DDP, calculation of metrics happen across all GPUs. 
    This function aggregate the loss across all GPUs. 
    """
    if not is_dist():
        return value
    all_loss = torch.tensor([value], device=device)
    dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)
    return all_loss.item() / dist.get_world_size()
    # return value

def init_print_override():
    '''
    Overriding the print function is useful when running DDP. 
    This way, only rank 0 prints to the console.
    '''
    import builtins as __builtin__
    
    original_print = __builtin__.print

    def print(*args, **kwargs):
        if os.getenv('GLOBAL_RANK') == '0':
            original_print(*args, **kwargs)

    __builtin__.print = print

    return original_print

def restore_print_override(original_print):
    '''
    Restore the original print function.
    '''
    import builtins as __builtin__
    __builtin__.print = original_print




# Function to print evaluation results and benchmark results
def print_evaluation_results(iter_num, eval_results, benchmark_results, text_modeling_results):
    headers = ['Metric', 'Value']
    table = PrettyTable(headers)

    # Adding eval_results rows
    for metric, value in eval_results.items():
        row = [metric, value]
        table.add_row(row)

    print(f"Iteration {iter_num}")
    print(table)

    
    benchmark_table = PrettyTable(['Benchmark', 'Accuracy', "Path Conf.", "Ground Conf."])
    for eval_method in benchmark_results.keys():
        if eval_method == "ft_qa":
            continue
        for benchmark, value in benchmark_results[eval_method].items():
            benchmark_table.add_row([
                f"{benchmark}", 
                value['accuracy'],
                value['path_confidence'],
                value['ground_confidence']
            ])

    print("Benchmark Results")
    print(benchmark_table)

    text_modeling_table = PrettyTable(['Topic', 'Difficulty', 'Norm. Lev. Dist.', 'Byte Acc.', 'Byte Perplexity'])
    for topic in text_modeling_results.keys():
        for difficulty, value in text_modeling_results[topic].items():
            text_modeling_table.add_row([
                f"{topic}", 
                f"{difficulty}",
                value['Norm. Lev. Dist.'],
                value['Byte Acc.'],
                value['Byte Perplexity']
            ])

    print("Text Modeling Results")
    print(text_modeling_table)



