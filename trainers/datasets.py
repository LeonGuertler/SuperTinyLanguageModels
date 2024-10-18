"""
A collection of dataloaders
"""

import os

import numpy as np
import torch
from tqdm import tqdm
import random

from models.embedding_models import GenericEmbedder
from trainers.data_utils import load_data, get_preprocessed_data_path



class DatasetInterface(torch.utils.data.IterableDataset):
    """
    A basic interface to be used by the remaining datasets
    """
    def __init__(self, split, cfg):
        """
        Arguments:
            cfg: the train script cfg
        """
        super().__init__()
        self.cfg = cfg
        # dataset_names = self.cfg["trainer"]["dataset_names"]
        self.context_window = self.cfg["model"]["context_window"]
        
        # # Create a unique identifier for the combined datasets
        self.data_path = os.path.join(
            get_preprocessed_data_path(cfg),
            f"{split}.bin"
        )
        
        self._load_data()
        self.dataset_len = len(self.data) - self.context_window


    def _load_data(self):
        """
        Get data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist, preprocess the data first")
        self.data = np.memmap(
            self.data_path,
            dtype=np.uint16,
            mode="r",
        )

    def __len__(self):
        """
        Return dataset length
        """
        return self.dataset_len
    
    def __iter__(self, idx):
        raise NotImplementedError
    
    
class BaseDatasetRandom(DatasetInterface):
    """
    Simple base dataloader for standard gpt-2'esk architectures and training.
    """
    def __init__(self, split, cfg):
        super().__init__(split, cfg)

    
    def __iter__(self):
        """
        Get a batch of random data points in an infinite loop.
        """
        while True:
            # Get a random index
            idx = random.randint(0, self.dataset_len - 1) 
        
            # Extract a slice of data for x and y
            x = torch.from_numpy((self.data[idx: idx + self.context_window]).astype(np.int64))
            y = torch.from_numpy((self.data[idx + 1: idx + 1 + self.context_window]).astype(np.int64))
            
            # Yield the data points
            yield x, y

class DocumentClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset class for document classification tasks.
    """

    def __init__(self, split, cfg):
        """
        Arguments:
            split: Dataset split ('train', 'validation', 'test')
            cfg: The training configuration dictionary
        """
        self.cfg = cfg
        self.split = split

        # Path to the preprocessed data
        data_folder = get_preprocessed_data_path(cfg)
        inputs_filename = os.path.join(data_folder, f"{split}_inputs.npy")
        labels_filename = os.path.join(data_folder, f"{split}_labels.npy")

        # Load the inputs and labels
        if not os.path.exists(inputs_filename) or not os.path.exists(labels_filename):
            raise FileNotFoundError(
                f"Data files for split '{split}' not found. Please preprocess the data first."
            )

        self.inputs = np.load(inputs_filename, allow_pickle=True)
        self.labels = np.load(labels_filename)

        assert len(self.inputs) == len(self.labels), "Mismatch between inputs and labels length."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label


DATASET_REGISTRY = {
    "random": BaseDatasetRandom,
    "text_classification": DocumentClassificationDataset
}

def build_dataset(cfg, split):
    """ TODO """
    return DATASET_REGISTRY[cfg["trainer"]["dataloader_name"]](
        split=split,
        cfg=cfg
    )