"""
A collection of dataloaders
"""
import os 
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from trainers.utils import (
    load_data,
)




class StandardDataloader(torch.nn.Module):
    """
    A basic dataloader that preprocesses a dataset via 
    tokenization, and then randomly loads batches as 
    necessary.
    Args:
        cfg: the data config
        preprocess: whether to preprocess the data
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self, tokenizer):
        """
        Tokenize and store the data
        """

        # load the dataset
        split_dataset = load_data(
            dataset_name=self.cfg["dataset_name"],
        )

        def process(example):
            ids = tokenizer.encode_ordinary(example['text'])
            ids.append(tokenizer.eot_token)
            return {'ids': ids, 'len': len(ids)}
            
        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=8,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()