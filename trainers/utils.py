"""Utilities for the trainer"""

import importlib
from prettytable import PrettyTable
import inspect
import os
import pkgutil

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets

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



DATASET_DICT = {
    "debug": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "babylm_100m": lambda: load_dataset("Sree1994/babylm_100M"), # https://babylm.github.io/
    "tinystories": lambda: load_dataset("roneneldan/TinyStories"), # https://huggingface.co/datasets/roneneldan/TinyStories
    "stlm": create_stlm_data_mix,
    "openhermes-2.5": lambda: load_dataset("teknium/OpenHermes-2.5"),
    "openwebtext": lambda: load_dataset("Skylion007/openwebtext"),
    "github-code": lambda: load_github_code_dataset(),
    "competition_math": lambda: load_competition_math_dataset(),
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
def print_evaluation_results(iter_num, eval_results, benchmark_results):
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


