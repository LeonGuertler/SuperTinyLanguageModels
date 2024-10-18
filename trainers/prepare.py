"""
Data processing module for preparing datasets for training.
"""

import os
import numpy as np
from tqdm import tqdm
from trainers.data_utils import load_data, get_preprocessed_data_path
from models.build_models import build_embedding_model

# Base class for data processors
class BasePreProcessor:
    """
    Base class for data processors.
    Provides an interface to process data and write tokenized data to disk.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def process(self, example):
        """
        Tokenizes the input example.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        """
        Writes the tokenized data to disk.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class TextPreProcessor(BasePreProcessor):
    """ TODO """
    def __init__(self, embedder):
        super().__init__(embedder)

    def process(self, sample):
        ids = self.embedder.tokenize_input(sample["text"])
        return {"ids": ids, "len": len(ids)}

    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        """ TODO """
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(tokenized_data_folder, f"{split}.bin")
            dtype = np.uint16  # Assumes token IDs are less than 2**16
            arr = np.memmap(
                filename,
                dtype=dtype,
                mode="w+",
                shape=(arr_len,),
            )
            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into memory-mapped array
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush() 




# Dictionary mapping processor names to classes
PROCESSORS = {
    "text_preprocessor": TextPreProcessor,
}


def prepare_data(cfg):
    """ TODO """
    # Extract configuration parameters
    preprocessor_name = cfg["trainer"]["preprocessor_name"]
    dataset_names = cfg["trainer"]["dataset_names"]

    # Ensure dataset_names is a list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Define the tokenized data folder path
    tokenized_data_folder = get_preprocessed_data_path(cfg)

    # Check if the data is already processed
    if os.path.exists(tokenized_data_folder) and len(os.listdir(tokenized_data_folder)) != 0:
        print("Tokenized data already exists.")
        return
    else:
        os.makedirs(tokenized_data_folder, exist_ok=True)


    # Load embedder
    embedder = build_embedding_model(cfg["model"])

    # Load the dataset
    split_dataset = load_data(dataset_names=dataset_names)

    # Initialize the processor
    processor_class = PROCESSORS.get(preprocessor_name)
    if processor_class is None:
        raise ValueError(f"Processor '{preprocessor_name}' not recognized.")
    processor = processor_class(embedder=embedder)

    try:
        # Determine the number of processors to use
        max_procs = min(os.cpu_count(), 12)
        print(f"Using {max_procs} processes for tokenization.")

        # Tokenize the dataset
        tokenized = split_dataset.map(
            processor.process,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=max_procs,
        )

        # Write tokenized data to disk
        processor.write_tokenized_data(
            tokenized=tokenized,
            tokenized_data_folder=tokenized_data_folder,
        )

    except Exception as exc:
        print(f"Error during data processing: {exc}")
        # Clean up partial files
        for file in os.listdir(tokenized_data_folder):
            os.remove(os.path.join(tokenized_data_folder, file))
        raise RuntimeError("Failed to process and write data.") from exc
