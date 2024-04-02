"""Utilities for the trainer"""
import os 

import torch
import numpy as np

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
    "debug": lambda: load_dataset("karpathy/tiny_shakespeare"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
}

def load_data(dataset_name, shuffle=True):
    """Load the data"""
    assert dataset_name in DATASET_DICT, f"Dataset {dataset_name} not found!"
    dataset = DATASET_DICT[dataset_name]()

    # create dataset split
    split_dataset = dataset["train"].train_test_split(
        test_size=0.01,
        seed=489,
        shuffle=shuffle
    )

    # rename test split to val
    split_dataset["val"] = split_dataset.pop("test")

    # return the training and validation datasets
    return split_dataset

