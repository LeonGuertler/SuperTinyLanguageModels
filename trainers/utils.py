""" General trainer utils """

import importlib
import inspect
import os
import pkgutil
import numpy as np
from prettytable import PrettyTable

import torch 
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
def print_evaluation_results(iter_num, eval_results):
    val_table = PrettyTable(["Metric", "Value"])
    mcq_table = PrettyTable(["Benchmark", "Accuracy"])
    text_modeling_table = PrettyTable(
        [
            "Topic", "Difficulty", "Byte Acc.", 
            "Byte Lev. Dist.", "Byte Perplexity"
        ]
    )
    text_generation_table = PrettyTable(
        [
            "Metric", "Value"
        ]
    )

    text_modeling_struct = {}
    for eval_name in eval_results.keys():
        if "Validation" in eval_name:
            val_table.add_row(
                [eval_name, eval_results[eval_name]]
            )
        elif "Text Modeling" in eval_name:
            metric = eval_name.split(")/")[0].replace(
                "Text Modeling (", ""
            )
            category = eval_name.split("/")[1].split("-")[0]
            difficulty = eval_name.split("/")[1].split("-")[1]
            if category not in text_modeling_struct:
                text_modeling_struct[category] = {}
            if difficulty not in text_modeling_struct[category]:
                text_modeling_struct[category][difficulty] = {}
            text_modeling_struct[category][difficulty][metric] = eval_results[eval_name]
        elif "MCQ" in eval_name:
            mcq_table.add_row(
                [eval_name, eval_results[eval_name]]
            )
        elif "Text Generation" in eval_name:
            text_generation_table.add_row(
                [eval_name.replace('Text Generation/',''), eval_results[eval_name]]
            )
        elif eval_name in ["iter", "token_num"]:
            continue # skip these
        else:
            print(f"Eval pretty print received: {eval_name} metric without printing it. It'll still be logged in wandb")

    # populate text modeling table
    for category in text_modeling_struct.keys():
        for difficulty in text_modeling_struct[category].keys():
            text_modeling_table.add_row(
                [
                    category, 
                    difficulty,
                    text_modeling_struct[category][difficulty]["Byte Acc."],
                    text_modeling_struct[category][difficulty]["Byte Lev. Dist."],
                    text_modeling_struct[category][difficulty]["Byte Perplexity"]
                ]
            )


    print(f"Token Num: {eval_results['token_num']}\tIteration {iter_num} - Validation Results")
    print(val_table)
    print(f"\n MCQ Results")
    print(mcq_table)
    print(f"\n Text-Modeling Results")
    print(text_modeling_table)
    print(f"\n Text-Generation Results")
    print(text_generation_table)
