"""
A collection of dataloaders
"""

import os

import numpy as np
import torch
from tqdm import tqdm

from models.embedding_models import GenericEmbedder
from trainers.utils import load_data


class BaseDataloader:
    """Abstract class for dataloaders"""

    def __init__(
        self,
        cfg,
        embedder,
    ):
        """Arguments:
        cfg: the train script cfg,
        tokenizer: the tokenizer object
            This is required to pre-tokenize the data
        """
        self.model_cfg = cfg.model
        self.trainer_cfg = cfg.trainer
        self.tokenized_data_dir = cfg["general"]["paths"]["data_dir"]
        self.embedder: GenericEmbedder = embedder
        self.context_window = self.model_cfg["context_window"]
        self.vocab_size = self.model_cfg["vocab_size"]
        self.batch_size = self.trainer_cfg["training"]["batch_size"]
        self.dataset_name = self.trainer_cfg["dataset"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenized_data_path = os.path.join(
            self.tokenized_data_dir,
            self.dataset_name,
            f'{self.model_cfg["embedder"]["tokenizer_type"]}-{self.model_cfg["vocab_size"]}',
        )

    def get_batch(self, split="train"):
        """
        Get a train/val batch
        """
        data = np.memmap(
            os.path.join(self.tokenized_data_path, f"{split}.bin"),
            dtype=np.uint16,
            mode="r",
        )

        idxs = torch.randint(len(data) - self.context_window, (self.batch_size,))
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + self.context_window]).astype(np.int64))
                for i in idxs
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.context_window]).astype(np.int64)
                )
                for i in idxs
            ]
        )

        X, y = X.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
            self.device, non_blocking=True
        )

        return X, y

    def check_processed(self):
        """
        Check if the data has been preprocessed
        """
        return os.path.exists(self.tokenized_data_path)

    def _write_tokenized_data(self, tokenized):
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(self.tokenized_data_path, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    def prepare_data(self):
        """
        Tokenize and store the data
        """
        # create folder
        if not os.path.exists(self.tokenized_data_path):
            os.makedirs(self.tokenized_data_path)
        else:
            return  # already processed

        # load the dataset
        split_dataset = load_data(
            dataset_name=self.dataset_name,
        )

        def process(example):
            ids = self.embedder.tokenize_input(example["text"])
            return {"ids": ids, "len": len(ids)}

        try:
            # tokenize the dataset
            tokenized = split_dataset.map(
                process,
                remove_columns=["text"],
                desc="tokenizing the splits",
                num_proc=1,
            )

            # concatenate all the ids in each dataset into one large file we can use for training
            self._write_tokenized_data(tokenized)
        except RuntimeError as exc:
            # if we fail, destroy the file
            os.removedirs(self.tokenized_data_path)
            raise SystemExit from exc


class StandardDataloader(BaseDataloader):
    """
    A basic dataloader that preprocesses a dataset via
    tokenization, and then randomly loads batches as
    necessary.
    """


class Seq2SeqDataloader(BaseDataloader):
    """
    A sequence to sequence dataloader that preprocess a dataset
    via tokenization, and then randomly loads batches of
    X,y pairs where both X and y are sequences of tokens of
    a specific length
    Args:
        cfg: the data config
        preprocess: whether to preprocess the data
    """

    def get_batch(self, split="train"):
        """
        Get a train/val batch
        """
        data = np.memmap(
            os.path.join(self.tokenized_data_path, f"{split}.bin"),
            dtype=np.uint16,
            mode="r",
        )

        idxs = torch.randint(len(data) - 2 * self.context_window, (self.batch_size,))
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + self.context_window]).astype(np.int64))
                for i in idxs
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (
                        data[i + self.context_window : i + 2 * self.context_window]
                    ).astype(np.int64)
                )
                for i in idxs
            ]
        )

        X, y = X.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
            self.device, non_blocking=True
        )

        return X, y


class BytePoolingDataloader(BaseDataloader):
    """
    A basic dataloader that preprocesses a dataset via
    tokenization, and then randomly loads batches as
    necessary.
    Args:
        cfg: the data config
        preprocess: whether to preprocess the data
    """

    def __init__(self, cfg, embedder):
        super().__init__(cfg, embedder=embedder)
        self.tokenized_data_path += f"-BytePooling"

    def _write_tokenized_data(self, tokenized):
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(self.tokenized_data_path, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(
                filename,
                dtype=dtype,
                mode="w+",
                shape=(arr_len, self.model_cfg.embedder.byte_context_window),
            )
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                input(arr_batch.size())
                print(arr_batch.shape)
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
