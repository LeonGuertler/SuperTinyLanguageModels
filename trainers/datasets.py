"""
A collection of dataloaders
"""

import enum
import os

import numpy as np
import pydantic
import torch

from high_level_configs import GeneralConfig
from models.experimental.byte_level import byte_model_shell
from models.model_shell import ModelShellConfig
from trainers.config import TrainerConfig
from trainers.utils import DatasetEnum


class DatasetTypeNames(str, enum.Enum):
    """Possible dataset types"""

    STANDARD = "standard"
    BYTE_POOLED = "byte_pooling"
    DUAL_BYTE_POOLED = "dual_byte_pooling"


class DatasetConfig(pydantic.BaseModel):
    """Configuration for the dataset"""

    dataset_type: DatasetTypeNames
    dataset: DatasetEnum


class BaseDatasetConfig(pydantic.BaseModel):
    """Configuration for dataset specifying the type"""

    dataset: DatasetEnum
    dataset_type: DatasetTypeNames.STANDARD


class BytePoolingDatasetConfig(pydantic.BaseModel):
    """Configuration for BytePooling dataset"""

    dataset: DatasetEnum
    dataset_type: DatasetTypeNames.BYTE_POOLED


class DualBytePoolingDatasetConfig(pydantic.BaseModel):
    """Configuration for DualBytePooling dataset"""

    dataset: DatasetEnum
    dataset_type: DatasetTypeNames.DUAL_BYTE_POOLED


class DatasetInterface(torch.utils.data.Dataset):
    """
    A basic interface to be used by the remaining datasets
    """

    def __init__(
        self,
        split,
        dataset_cfg: DatasetConfig,
        model_cfg: ModelShellConfig,
        trainer_cfg: TrainerConfig,
        general_cfg: GeneralConfig,
    ):
        """
        Arguments:
            cfg: the train script cfg
        """
        super().__init__()
        self.dataset_name = dataset_cfg.dataset
        self.context_window = model_cfg.context_window
        self.data_path = os.path.join(
            general_cfg.paths.data_dir,
            self.dataset_name,
            (
                f"{model_cfg.embedding_model.tokenizer_type}"
                f"-{model_cfg.vocab_size}-{trainer_cfg.dataset}"
            ),
            f"{split}.bin",
        )

        self._load_data()
        self.dataset_len = len(self.data) - self.context_window

    def _load_data(self):
        """
        Get data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"{self.data_path} does not exist, preprocess the data first"
            )
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


class BaseDataset(DatasetInterface):
    """
    Simple base dataloader for standard gpt-2'esk architectures and training.
    """

    def __getitem__(self, idx):
        """
        Get a batch of data
        """
        x = torch.from_numpy(
            (self.data[idx : idx + self.context_window]).astype(np.int64)
        )
        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
        )
        return x, y


class BytePoolingDataset(DatasetInterface):
    """
    Simple byte-level dataset
    """

    def __init__(
        self,
        split,
        dataset_cfg: DatasetConfig,
        model_cfg: byte_model_shell.ByteShellConfig,
        training_cfg: TrainerConfig,
        general_cfg: GeneralConfig,
    ):
        super().__init__(split, dataset_cfg, model_cfg, training_cfg, general_cfg)
        self.loading_shape = None
        self._load_data()
        self.model_cfg = model_cfg

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
            self.loading_shape = (
                len(data) // self.model_cfg.byte_context_window,
                self.model_cfg.byte_context_window,
            )
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
        x = torch.from_numpy(
            (self.data[idx : idx + self.context_window]).astype(np.int64)
        )
        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
        )
        return x, y


class DualBytePooling(DatasetInterface):
    """
    Dataset for both byte-level and higher token level tokens simultaneously
    """

    def __init__(
        self,
        split,
        dataset_cfg: DatasetConfig,
        model_cfg: byte_model_shell.ByteShellConfig,
        training_cfg: TrainerConfig,
        general_cfg: GeneralConfig,
    ):
        self.loading_shape = None
        # overwrite datapath
        data_folder = os.path.join(
            general_cfg.paths.data_dir,
            dataset_cfg.dataset,
            (
                f"{model_cfg.embedding_model.tokenizer_type}"
                f"-{model_cfg.vocab_size}-{training_cfg.dataset}"
            ),
        )
        self.data_path_byte = os.path.join(data_folder, f"{split}_byte.bin")
        self.data_path_token = os.path.join(data_folder, f"{split}_token.bin")
        super().__init__(split, dataset_cfg, model_cfg, training_cfg, general_cfg)
        # force parent init
        self._load_data()
        self.model_cfg = model_cfg

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
            self.loading_shape = (
                len(data) // self.model_cfg.byte_context_window,
                self.model_cfg.byte_context_window,
            )
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

    def __getitem__(self, idx):
        """
        Get a batch of data from both the byte and higher token level
        """
        # get byte level batch
        x_byte = torch.from_numpy(
            (self.data_byte[idx : idx + self.context_window]).astype(np.int64)
        )
        # y_byte = torch.from_numpy(
        # (self.data_byte[idx + 1: idx + 1 + self.context_window]).astype(np.int64)
        # )

        # get token level batch
        # x_token = torch.from_numpy(
        # (
        # self.data_token[idx: idx + self.context_window]
        # ).astype(np.int64))
        y_token = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
        )
        return x_byte, y_token
