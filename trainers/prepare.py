"""
Necessary to be run before training to make sure all of the data is preprcessed etc.
"""
import os 
import torch 
import numpy as np 
from tqdm import tqdm 
from trainers.utils import load_data


class StandardProcessor:
    """
    A standard processor that tokenizes the text
    """
    def __init__(self, embedder):
        self.embedder = embedder
    def process(self, example):
        ids = self.embedder.tokenize_input(example["text"])
        return {"ids": ids, "len": len(ids)}
    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        """
        Write the tokenized data to a file
        """
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(tokenized_data_folder, f"{split}.bin")
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
    


DATALOADER_PROCESSORS = {
    "standard": StandardProcessor
}





def prepare_data(self, cfg, embedder):
    """
    Split the data, process & tokenize it, and store 
    it as memmap bin files
    """
    # check if the data is already preprocessed
    dataloader_name = cfg["trainer"]["dataloader"]
    dataset_name = cfg["general"]["dataset_name"]
    tokenized_data_folder = os.path.join(
        cfg["general"]["paths"]["data_dir"],
        dataset_name,
        f'{cfg["model"]["embedder"]["tokenizer_type"]}-{cfg["model"]["vocab_size"]}',
    )

    # check if train.bin already exists
    if (
        os.path.exists(os.path.join(tokenized_data_folder, "train.bin")) and 
        os.path.exists(os.path.join(tokenized_data_folder, "val.bin"))
    ):
        print("Tokenized data already exists")
        return
    if (
        os.path.exists(os.path.join(tokenized_data_folder, "train.bin")) or 
        os.path.exists(os.path.join(tokenized_data_folder, "val.bin"))
    ):
        # for now just delete and tokenized both again
        print("Deleting half-complete tokenized data")
        for split in ["train", "val"]:
            os.remove(os.path.join(tokenized_data_folder, f"{split}.bin"))
    
    # create the folder if it doesn't exist   
    if not os.path.exists(tokenized_data_folder):
        os.makedirs(tokenized_data_folder)

    # load the dataset
    split_dataset = load_data(
        dataset_name=dataset_name,
    )

    processor_object = DATALOADER_PROCESSORS[dataloader_name](
        embedder=embedder
    )

    # wrap in try such that half-complete files can be deleted on error
    try:
        # tokenize the dataset
        tokenized = split_dataset.map(
            processor_object.process,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=1
        )

        # concatenate all the ids in each dataset
        processor_object.write_tokenized_data(
            tokenized=tokenized, 
            tokenized_data_folder=tokenized_data_folder
        )

    except Exception as exc:
        for split in tokenized.keys():
            os.remove(os.path.join(self.tokenized_data_path, f"{split}.bin"))
        raise RuntimeError("Failed to process and write data") from exc


