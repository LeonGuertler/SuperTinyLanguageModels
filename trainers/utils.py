"""Utilities for the trainer"""

import os

import numpy as np
import torch
from datasets import load_dataset


def set_seed(seed):
    """Setup the trainer"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def create_folder_structure(path_config):
    """
    Create all necessary folders for training.
    """
    if not os.path.exists(path_config["data_path"]):
        os.makedirs(path_config["data_path"])

    if not os.path.exists(path_config["checkpoint_dir"]):
        os.makedirs(path_config["checkpoint_dir"])


DATASET_DICT = {
    "debug": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
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


def profilize(model, module_name=None):
    """Recursively add hooks to the model for recording PyTorch profiler traces with module names"""
    for idx, module in enumerate(model.children()):
        if isinstance(module, torch.nn.Module):
            child_module_name = f"{module_name}.{idx}" if module_name else str(idx)
            profilize(module, child_module_name)

    if hasattr(model, "forward"):

        def forward_wrapper(*args, **kwargs):
            try:
                nested_module_name = module_name or model.__class__.__name__
                torch.ops.profiler.record_function(f"{nested_module_name}.forward")
                output = model.forward(*args, **kwargs)
            finally:
                torch.ops.profiler.record_function("# End: {module_name}.forward")
            return output

        model.forward = forward_wrapper
