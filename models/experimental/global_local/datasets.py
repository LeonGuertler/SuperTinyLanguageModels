import os 
import torch 
import random
import numpy as np 
from trainers.data_utils import load_data, get_preprocessed_data_path



class DualBaseDatasetRandom(torch.utils.data.IterableDataset):
    def __init__(self, split, cfg):
        # load the data
        self.cfg = cfg 
        self.context_window = cfg["model"]["context_window"]

        self.data_path_1 = os.path.join(
            get_preprocessed_data_path(cfg),
            f"{split}_byte_ids.bin"
        ) 

        self.data_path_2 = os.path.join(
            get_preprocessed_data_path(cfg),
            f"{split}_delimitations.bin"
        ) 

        self.data_1 = np.memmap(
            self.data_path_1,
            dtype=np.uint16,
            mode="r"
        )

        self.data_2 = np.memmap(
            self.data_path_2,
            dtype=np.uint16,
            mode="r"
        )

        self.dataset_len = len(self.data_1) - self.context_window
    
    def __len__(self):
        return self.dataset_len  

    def __iter__(self):
        while True:
            # Get a random index
            idx = random.randint(0, self.dataset_len - 1) 
        
            # Extract a slice of data for x and y
            x = torch.from_numpy((self.data_1[idx: idx + self.context_window]).astype(np.int64))
            y = torch.from_numpy((self.data_1[idx + 1: idx + 1 + self.context_window]).astype(np.int64))
            
            delimitations = torch.from_numpy((self.data_2[idx: idx + self.context_window]).astype(np.int64))
            # Yield the data points
            yield x, delimitations, y