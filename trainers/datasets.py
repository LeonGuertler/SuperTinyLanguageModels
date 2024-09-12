"""
A collection of dataloaders
"""

import os

import numpy as np
import torch
from tqdm import tqdm
import random

from models.embedding_models import GenericEmbedder
from trainers.data_utils import load_data



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
        self.dataset_name = self.cfg["trainer"]["dataset"]
        self.context_window = self.cfg["model"]["context_window"]
        self.data_path = os.path.join(
            self.cfg["general"]["paths"]["data_dir"],
            self.dataset_name,
            f'{self.cfg["model"]["embedder"]["tokenizer_type"]}-{self.cfg["model"]["vocab_size"]}-{self.cfg["trainer"]["dataloader"]["name"]}',
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


class BytePoolingDataset(DatasetInterface):
    """
    Simple byte-level dataset
    """
    def __init__(self, split, cfg):
        self.loading_shape = None
        super().__init__(split, cfg)
        # force parent init
        self._load_data()

    def _load_data(self):
        """
        Get data
        """
        if self.loading_shape is None:
            data = np.memmap(
                self.data_path,
                dtype=np.uint16,
                mode="r",
            )
            self.loading_shape = (len(data)// self.cfg["model"]["embedder"]["byte_context_window"], self.cfg["model"]["embedder"]["byte_context_window"])
            data = None
        self.data = np.memmap(
            self.data_path,
            dtype=np.uint16,
            mode="r",
            shape=self.loading_shape,
        )
    
    def __iter__(self):
        """
        Get a batch of data
        """
        while True:
            idx = random.randint(0, self.dataset_len - 1)
            x = torch.from_numpy((self.data[idx: idx + self.context_window]).astype(np.int64))
            y = torch.from_numpy((self.data[idx + 1: idx + 1 + self.context_window]).astype(np.int64))
            yield x, y
    

class DualBytePooling(DatasetInterface):
    """
    Dataset for both byte-level and higher token level tokens simultaneously
    """
    def __init__(self, split, cfg):
        self.loading_shape = None
        # overwrite datapath
        data_folder = os.path.join(
            cfg["general"]["paths"]["data_dir"],
            cfg["trainer"]["dataset"],
            f'{cfg["model"]["embedder"]["tokenizer_type"]}-{cfg["model"]["vocab_size"]}-{cfg["trainer"]["dataloader"]["name"]}',
        )
        self.data_path_byte = os.path.join(data_folder, f"{split}_byte.bin")
        self.data_path_token = os.path.join(data_folder, f"{split}_token.bin")
        super().__init__(split, cfg)

        # force parent init
        self._load_data()

    def _load_data(self):
        """
        Get both the byte-level and the token level data
        """
        if self.loading_shape is None:
            data = np.memmap(
                self.data_path_byte,
                dtype=np.uint16,
                mode="r",
            )
            self.loading_shape = (len(data)// self.cfg["model"]["embedder"]["byte_context_window"], self.cfg["model"]["embedder"]["byte_context_window"])
            data = None
        self.data_byte = np.memmap(
            self.data_path_byte,
            dtype=np.uint16,
            mode="r",
            shape=self.loading_shape,
        )
        self.data = np.memmap(
            self.data_path_token,
            dtype=np.uint16,
            mode="r",
        )
    
    def __iter__(self):
        """
        Get a batch of data from both the byte and higher token level
        """
        while True:
            idx = random.randint(0, self.dataset_len - 1)
            # get byte level batch
            x_byte = torch.from_numpy((self.data_byte[idx: idx + self.context_window]).astype(np.int64))
            #y_byte = torch.from_numpy((self.data_byte[idx + 1: idx + 1 + self.context_window]).astype(np.int64))

            # get token level batch
            #x_token = torch.from_numpy((self.data_token[idx: idx + self.context_window]).astype(np.int64))
            y_token = torch.from_numpy((self.data[idx + 1: idx + 1 + self.context_window]).astype(np.int64))
            yield x_byte, y_token  

