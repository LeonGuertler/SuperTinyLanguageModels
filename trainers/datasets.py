"""
A collection of dataloaders
"""

import os

import numpy as np
import torch
from tqdm import tqdm
import random

from models.embedding_models import GenericEmbedder
from trainers.utils import load_data



class DatasetInterface(torch.utils.data.Dataset):
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
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def check_processed(self):
        """
        Check if the data has been preprocessed
        """
        # raise error if not exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist")
        


class BaseDataset(DatasetInterface):
    """
    Simple base dataloader for standard gpt-2'esk architectures and training.
    """
    def __init__(self, split, cfg):
        super().__init__(split, cfg)

    
    def __getitem__(self, idx):
        """
        Get a batch of data
        """
        x = torch.from_numpy((self.data[idx: idx + self.context_window]).astype(np.int64))
        y = torch.from_numpy((self.data[idx + 1: idx + 1 + self.context_window]).astype(np.int64))
        return x, y


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

    
    def __getitem__(self, idx):
        """
        Get a batch of data
        """
        x = torch.from_numpy((self.data[idx: idx + self.context_window]).astype(np.int64))
        y = torch.from_numpy((self.data[idx + 1: idx + 1 + self.context_window]).astype(np.int64))
        return x, y


class BytePoolingAutoencodingDataset(BytePoolingDataset):
    """
    Slight variation of the BytePoolingDataset that introduces a Autoencoding component
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        """
        Get a batch of data
        """
        x = torch.from_numpy((self.data[idx: idx + self.context_window]).astype(np.int64))
        y = x
        return x, y
      