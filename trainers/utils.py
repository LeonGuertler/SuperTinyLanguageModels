""" General trainer utils """

import importlib
import inspect
import os, re, wandb
import pkgutil
import numpy as np
from prettytable import PrettyTable
from collections import defaultdict
from typing import Optional, List, Dict, Any

import torch 
import torch.distributed as dist

import evals
from tqdm import tqdm

def intra_training_evaluation(model, benchmarks):
    """
    Evaluates the model on multiple benchmarks during training.
    
    Args:
        model: The model to evaluate.
        benchmarks List[str]: A list of benchmark names to evaluate the model on.
    """
    if benchmarks is None:
        return []
    results_list = []

    # Outer progress bar for benchmarks
    with tqdm(benchmarks, desc="Evaluating benchmarks", position=0,  leave=True) as benchmark_bar:
        for benchmark in benchmark_bar:
            try:
                # Create the benchmark evaluator
                benchmark_evaluator = evals.make(benchmark)

                # Evaluating within the benchmark, tqdm already exists in the yield function
                results = benchmark_evaluator.evaluate(model=model)
                results_list.append(results)
            except Exception as e:
                print(f"[EXCEPTION] during intra_training_evaluation on {benchmark}. \nDetails:{e}")
    return results_list

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


def print_evaluation_results(eval_results: Dict[str, Any], benchmark_results: List[Dict[str, Any]]):
    """
    Processes and visualizes the evaluation results and benchmark results.

    Args:
        eval_results (dict): Dictionary containing evaluation metrics and other related information.
        benchmark_results (list of dict): List of dictionaries containing benchmark results.
    """
    # Helper function to format numerical values
    def format_value(metric_name: str, value: Any) -> str:
        if isinstance(value, float):
            if 'accuracy' in metric_name.lower():
                # Format as percentage with two decimal places
                return f"{value * 100:.2f}%"
            else:
                return f"{value:.4g}"
        return value

    # Process and print evaluation results
    if eval_results:
        # Keys to be ignored (e.g., 'token_num', 'iter')
        ignore_keys = set(["token_num", "iter"])

        # Filter out keys that don't have a "/" and are not in ignore_keys
        valid_keys = {
            k: v for k, v in eval_results.items()
            if "/" in k and k.split("/")[0] not in ignore_keys
        }

        if valid_keys:
            # Identify unique logging paths (the part before "/")
            logging_paths = set(k.split("/")[0] for k in valid_keys)

            # Dictionary to store tables for each logging path
            tables = {}

            # Loop through each logging path and generate a table
            for table_name in logging_paths:
                # Collect columns for the table (the part after "/")
                columns = sorted(
                    set(k.split("/")[1] for k in valid_keys if k.startswith(f"{table_name}/"))
                )

                # Initialize a table with the logging path as the category and columns for the metrics
                table = PrettyTable(["Evaluation"] + columns)
                table.align = "l"

                # Collect values for the current logging path
                row_values = {}
                for col in columns:
                    key = f"{table_name}/{col}"
                    value = valid_keys.get(key, "N/A")
                    row_values[col] = format_value(col, value)

                # Add the row to the table
                table.add_row([table_name] + [row_values[col] for col in columns])
                tables[table_name] = table

            # Print all evaluation tables
            for table_name, table in tables.items():
                print(f"\nResults for {table_name}:")
                print(table)
        else:
            print("No valid evaluation results to display.")
    else:
        print("No evaluation results provided.")

    # Process and print benchmark results
    if benchmark_results:
        # Group benchmarks by their type
        benchmarks_by_type = defaultdict(list)
        for benchmark in benchmark_results:
            benchmark_type = benchmark.get("benchmark_type", "Unknown Type")
            benchmarks_by_type[benchmark_type].append(benchmark)

        # Iterate through each benchmark type and create tables accordingly
        for benchmark_type, benchmarks in benchmarks_by_type.items():
            # Collect all unique metrics for this benchmark type
            all_metrics = set()
            for benchmark in benchmarks:
                results = benchmark.get("results", {})
                all_metrics.update(results.keys())

            # Remove 'Generated Text (html)' if present
            all_metrics.discard("Generated Text (html)")

            if not all_metrics:
                print(f"\nBenchmark Type: {benchmark_type}")
                print("No numerical metrics to display.")
                continue

            # Sort metrics for consistent column ordering
            sorted_metrics = sorted(all_metrics)

            # Initialize PrettyTable with Benchmark Name and dynamic metrics
            table = PrettyTable(["Benchmark Name"] + sorted_metrics)
            table.align = "l"

            for benchmark in benchmarks:
                benchmark_name = benchmark.get("benchmark_name", "Unnamed Benchmark")
                results = benchmark.get("results", {})
                # Prepare row with formatted values or 'N/A' if metric is missing
                row = [benchmark_name]
                for metric in sorted_metrics:
                    value = results.get(metric, "N/A")
                    formatted_value = format_value(metric, value)
                    row.append(formatted_value)
                table.add_row(row)

            # Print the benchmark table with its type as header
            print(f"\nBenchmark Type: {benchmark_type}")
            print(table)
    else:
        print("\nNo benchmark results to display.")


def wandbify_evaluation_results(benchmark_results: List[Dict[str, Any]], convert_accuracy_to_percentage: bool = False) -> Dict[str, Any]:
    """
    Processes the benchmark_results list of dicts 
    to be logged in Weights & Biases (wandb), formatting keys based on the number of metrics.

    Args:
        benchmark_results (list of dict): List of dictionaries containing benchmark results.
        convert_accuracy_to_percentage (bool): 
            If True, multiplies accuracy metrics by 100 to represent them as percentages.

    Returns:
        dict: A flat dictionary formatted for wandb logging with keys structured based on metric count.
    """
    import wandb

    wandb_log_dict = {}

    for benchmark in benchmark_results:
        benchmark_type = benchmark.get("benchmark_type", "Unknown_Type")
        benchmark_name = benchmark.get("benchmark_name", "Unnamed_Benchmark")
        results = benchmark.get("results", {})
        num_metrics = len(results) - ("Generated Text (html)" in results)

        for metric, value in results.items():
            if metric == "Generated Text (html)":
                # Handle HTML content separately using wandb.Html
                key = f"{benchmark_type} / {benchmark_name} / {metric}"
                wandb_log_dict[key] = wandb.Html(value)
                continue  # Skip further processing for HTML content

            # Optionally convert accuracy metrics to percentages
            if convert_accuracy_to_percentage and "accuracy" in metric.lower() and isinstance(value, float):
                value = value * 100

            # Format the key based on the number of metrics
            # if num_metrics == 1:
            # Single Metric Format: "benchmark_type / benchmark_name (metric)"
            key = f"{benchmark_type} / {benchmark_name} ({metric.split('/', '')})"
            # else:
            #     # Multiple Metrics Format: "benchmark_type (benchmark_name) / metric"
            #     key = f"{benchmark_type} ({benchmark_name}) / {metric}"

            wandb_log_dict[key] = value

    return wandb_log_dict