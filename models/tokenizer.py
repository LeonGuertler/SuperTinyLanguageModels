import os, pickle
from tqdm import tqdm
import numpy as np
import tiktoken

import torch

# from models.utils import load_datasets
from models.utils import load_datasets

TOKENIZERS = {
    "gpt2": lambda: tiktoken.get_encoding("gpt2"),
}


class tokenizer:
    def __init__(self, config):
        self.tokenizer = TOKENIZERS[config["arch"]["tokenizer"]]()
        self.config = config
        self.dataset_path = os.path.join(
            self.config["paths"]["data_path"],
            self.config["training"]["dataset"],
            self.config["arch"]["tokenizer"],
        )

        self.context_window = self.config["arch"]["context_window"]
        self.batch_size = self.config["training"]["batch_size"]

    def encode_text(self, text, device):
        start_ids = self.tokenizer.encode(text)
        return torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_batch(self, split="train"):
        data = np.memmap(
            os.path.join(self.dataset_path, f"{split}.bin"), dtype=np.uint16, mode="r"
        )

        ix = torch.randint(len(data) - self.context_window, (self.batch_size,))
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + self.context_window]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.context_window]).astype(np.int64)
                )
                for i in ix
            ]
        )

        X, y = X.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
            self.device, non_blocking=True
        )
        return X, y

    def prepare_dataset(self):
        """
        First, check if the dataset has already been prepared and stored
        using the correct tokenizer. If not, prepare the dataset and store it.
        """

        # check if the dataset has already been prepared
        if not os.path.exists(self.dataset_path):
            # check if dataset folder exists
            dataset_folder = os.path.join(
                self.config["paths"]["data_path"], self.config["training"]["dataset"]
            )
            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)
            # load the dataset
            dataset = load_datasets(self.config["training"]["dataset"])

            def process(example):
                ids = self.tokenizer.encode_ordinary(example["text"])
                ids.append(self.tokenizer.eot_token)
                return {"ids": ids, "len": len(ids)}

            # tokenize dataset
            tokenized = dataset.map(
                process,
                remove_columns=["url", "title", "text"],
                desc="Tokenizing the dataset",
                num_proc=8,  # number workers
            )

            # create dataset path if doesn't exist
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)

            # concatenate all the ids in each dataset into one large file for training
            for split, dset in tokenized.items():
                arr_len = np.sum(dset["len"], dtype=np.uint64)
                filename = os.path.join(self.dataset_path, f"{split}.bin")
                dtype = (
                    np.uint16
                )  # possible since enc.max_token_value ==50256 is < 2**16
                arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
                total_batches = 1024

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                    batch = dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    ).with_format("numpy")
                    arr_batch = np.concatenate(batch["ids"])

                    # write into mmap
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()
