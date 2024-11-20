import os 
import numpy as np 
from models.components.tokenizers import build_tokenizer
from tqdm import tqdm


class DualTextPreProcessor:
    """ TODO """
    def __init__(self, embedder):
        self.embedder = embedder

        self.global_tokenizer = build_tokenizer(
            tokenizer_type="o200k_base",
            vocab_size=None, dataset_names=None, simplify=None, num_reserved_tokens=None
        )

    def process(self, sample):
        """
        1. Extract delimitations
        2. embedd bytes
        3. store both
        """
        global_tokens = self.global_tokenizer.encode(sample["text"])

        delimitations = []
        byte_tokens = []
        for global_token_id in global_tokens:
            # decode into str
            bts = self.embedder.tokenizer.encode(self.global_tokenizer.decode([global_token_id]))
            dels = [0]*(len(bts)-1)+[1]

            delimitations += dels 
            byte_tokens += bts

        # add eot token

        return {"ids": byte_tokens, "delimitations": delimitations, "len": len(byte_tokens)}



    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        """ TODO """
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename_1 = os.path.join(tokenized_data_folder, f"{split}_byte_ids.bin")
            filename_2 = os.path.join(tokenized_data_folder, f"{split}_delimitations.bin")
            dtype = np.uint16  # Assumes token IDs are less than 2**16
            arr_1 = np.memmap(
                filename_1,
                dtype=dtype,
                mode="w+",
                shape=(arr_len,),
            )
            arr_2 = np.memmap(
                filename_2,
                dtype=dtype,
                mode="w+",
                shape=(arr_len,),
            )
            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename_1} and {filename_2}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_1_batch = np.concatenate(batch["ids"])
                arr_2_batch = np.concatenate(batch["delimitations"])
                # Write into memory-mapped array
                arr_1[idx : idx + len(arr_1_batch)] = arr_1_batch
                arr_2[idx : idx + len(arr_2_batch)] = arr_2_batch
                idx += len(arr_1_batch)
            arr_1.flush() 
            arr_2.flush() 