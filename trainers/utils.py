"""Utilities for the trainer"""

import os

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf


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


def flatten_omegaconf(cfg):
    """
    Flatten the omegaconf config into a single-level dictionary.
    """
    return_dict = {}
    for k, v in OmegaConf.to_container(cfg, resolve=True).items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                return_dict[k2] = v2
        else:
            return_dict[k] = v
    return return_dict

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
