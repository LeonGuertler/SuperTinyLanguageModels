"""
Necessary to be run before training to make sure all of the data is preprcessed etc.
"""
import os 
import torch 
import numpy as np 
from tqdm import tqdm 
from trainers.utils import load_data

from models.build_models import build_embedding_model 


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
            arr = np.memmap(
                filename, 
                dtype=dtype, 
                mode="w+", 
                shape=(arr_len,)
            )
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
    
class ByteLevelProcessor(StandardProcessor):
    """
    A byte-level processor that tokenizes the text
    """
    def __init__(self, embedder):
        super().__init__(embedder)

    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(tokenized_data_folder, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(
                filename,
                dtype=dtype,
                mode="w+",
                shape=(arr_len, 12), #TODO remove hardcoding
            )
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

class DualByteLevelProcessor(StandardProcessor):
    """
    This preprocessor stores both the byte level structure and 
    the standard structure to enable the training of architectures
    with byte-level input, but standard token output.
    """
    def __init__(self, embedder):
        super().__init__(embedder)

    def process(self, example):
        byte_ids, token_ids = self.embedder.tokenize_input(example["text"], return_high_level=True)
        print(byte_ids.size(), token_ids.size())
        input()
        return {"byte_ids": byte_ids, "token_ids": token_ids, "len": len(token_ids)}
    
    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)

            filename_byte = os.path.join(tokenized_data_folder, f"{split}_byte.bin")
            filename_token = os.path.join(tokenized_data_folder, f"{split}_token.bin")

            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)

            arr_byte = np.memmap(
                filename_byte,
                dtype=dtype,
                mode="w+",
                shape=(arr_len, 12), #TODO remove hardcoding
            )

            arr_token = np.memmap(
                filename_token,
                dtype=dtype,
                mode="w+",
                shape=(arr_len,),
            )

            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename_byte} and {filename_token}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch_byte = np.concatenate(batch["byte_ids"])
                arr_batch_token = np.concatenate(batch["token_ids"])

                # write into mmap
                arr_byte[idx : idx + len(arr_batch_byte)] = arr_batch_byte
                arr_token[idx : idx + len(arr_batch_token)] = arr_batch_token
                idx += len(arr_batch_byte)

            arr_byte.flush()
            arr_token.flush()



DATALOADER_PROCESSORS = {
    "standard": StandardProcessor,
    "byte_pooling": ByteLevelProcessor,
    "dual_byte_pooling": DualByteLevelProcessor
}





def prepare_data(cfg):
    """
    Split the data, process & tokenize it, and store 
    it as memmap bin files
    """
    # check if the data is already preprocessed
    dataloader_name = cfg["trainer"]["dataloader"]["name"]
    dataset_name = cfg["trainer"]["dataset"]
    tokenized_data_folder = os.path.join(
        cfg["general"]["paths"]["data_dir"],
        dataset_name,
        f'{cfg["model"]["embedder"]["tokenizer_type"]}-{cfg["model"]["vocab_size"]}-{cfg["trainer"]["dataloader"]["name"]}',
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


    # load embedder
    embedder = build_embedding_model(cfg["model"])

    # load the dataset
    split_dataset = load_data(
        dataset_name=dataset_name,
    )

    processor_object = DATALOADER_PROCESSORS[dataloader_name](
        embedder=embedder
    )

    # wrap in try such that half-complete files can be deleted on error
    try:
        # Get the maximum number of processors
        max_procs = os.cpu_count()
        print(f"Using {max_procs} processors")

        # tokenize the dataset
        tokenized = split_dataset.map(
            processor_object.process,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=max_procs
        )

        # concatenate all the ids in each dataset
        processor_object.write_tokenized_data(
            tokenized=tokenized, 
            tokenized_data_folder=tokenized_data_folder
        )

    except Exception as exc:
        print(f"Error: {exc}")
        for file in os.listdir(tokenized_data_folder):
            os.remove(os.path.join(tokenized_data_folder, file))
        raise RuntimeError("Failed to process and write data") from exc


