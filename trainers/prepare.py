"""
Necessary to be run before training to make sure all of the data is preprcessed etc.
"""
import os 
import torch 
import numpy as np 
from tqdm import tqdm 
from trainers.data_utils import load_data

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


class PRM800KProcessor(StandardProcessor):
    """ TODO """
    def __init__(self, embedder):
        super().__init__(embedder)
        self.start_of_thought_token = 4999 #embedder.start_of_thought_token
        self.end_of_thought_token = 4998 #embedder.end_of_thought_token

        self.start_of_answer_token = 4997 #embedder.start_of_answer_token
        self.end_of_answer_token = 4996 #embedder.end_of_answer_token

        self.answer_pad_token = -99



    def process(self, example):
        """
        Create all possible completions with labels and return them.
        """
        input(example)
        question = example["question"]["problem"]
        question_ids = self.embedder.tokenize_input(question)
        return_list = [(question_ids, [self.answer_pad_token]*len(question_ids))]

        steps = example["steps"]
        for step in steps:
            tmp_list = []
            for current_ids, current_labels in return_list:
                for completion in step["completions"]:
                    completion_ids = self.embedder.tokenize_input(completion["text"])

                    # add start and end tokens to the completion
                    completion_ids = [self.start_of_thought_token] + completion_ids + [self.end_of_thought_token]
                    completion_labels = [-99] * (len(completion_ids)+1) + [completion["rating"]]

                    tmp_list.append(
                        (
                            current_ids + completion_ids,
                            current_labels + completion_labels
                        )
                    )

            input(tmp_list)
            return_list = tmp_list.copy()
        return [{"ids":x[0], "targets":x[1], "len":len(x[0])} for x in return_list] # for x in return_list}



# { "steps": [ 
#     { "completions": [ 
#         { "text": "7.8 minutes is the same as 7 minutes and 0.8 minutes.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "Right, and since there are 60 seconds in a minute, then there are 60 * 7 = 420 seconds in 7 minutes.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "And since there are 60 seconds in a minute, then there are 60 * 0.8 = 48 seconds in 0.8 minutes.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "So, in total, there are 420 + 48 = 468 seconds in 7.8 minutes.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "Right. Let's check our work. 7.8 minutes is the same as 7 minutes and 0.8 minutes.", "rating": 0, "flagged": false }, 
#         { "text": "Exactly.\n\n# Answer\n\n468", "rating": 1, "flagged": false }, 
#         { "text": "That's correct.\n\n# Answer\n\n468", "rating": 1, "flagged": false }, 
#         { "text": "Correct.\n\n# Answer\n\n468", "rating": 1, "flagged": false }, 
#         { "text": "That's correct.\n\n# Answer\n\n468", "rating": 1, "flagged": false }, 
#         { "text": "Correct.\n\n# Answer\n\n468", "rating": 1, "flagged": false }, 
#         { "text": "That's right!\n\n# Answer\n\n468", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 1 } 
# ], "total_time": 92405, "finish_reason": "solution" }


# { "steps": [ 
#     { "completions": [ 
#         { "text": "Let's call our two-digit integers x.", "rating": 0, "flagged": false }, 
#         { "text": "Let's first think about the remainders when we divide by 8.", "rating": 0, "flagged": false }, 
#         { "text": "So we need to find the number of positive two-digit integers that are 2 more than a multiple of 8.", "rating": 1, "flagged": false }, 
#         { "text": "So we're looking for numbers that are two more than a multiple of 8.", "rating": 0, "flagged": false }, 
#         { "text": "So we have to find the number of integers that are two more than a multiple of 8.", "rating": 0, "flagged": false }, 
#         { "text": "Let's write out the first few multiples of 8.", "rating": 0, "flagged": false }, 
#         { "text": "So if a number leaves a remainder of 2 when divided by 8, it's of the form 8n+2.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 6 }, 
#     { "completions": [ 
#         { "text": "So we want to know the number of positive two-digit integers of the form 8n+2.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "I think we should just plug in numbers and see what happens.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "Ok let's start with n=1.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "8*1+2=10 which is a two-digit integer.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "Let's try n=2.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "8*2+2=18 which is also a two-digit integer.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "And if we keep going we'll see that all the numbers of the form 8n+2 are two-digit integers.", "rating": -1, "flagged": false }, 
#         { "text": "Let's try n=3.", "rating": 0, "flagged": false }, 
#         { "text": "Let's try n=3.", "rating": 0, "flagged": false }, 
#         { "text": "Let's try n=3.", "rating": 0, "flagged": false }, 
#         { "text": "3, 4, 5 and 6 work as well.", "rating": 0, "flagged": false }, 
#         { "text": "But if n=3, then 8*3+2=26 which is a three-digit integer.", "rating": 0, "flagged": false }, 
#         { "text": "And if we keep going like this, we'll see that every integer of the form 8n+2 is a two-digit integer.", "rating": 0, "flagged": false } 
#         ], "human_completion": {
#              "text": "And let's keep plugging numbers until we get a three-digit number.", "rating": null, "source": "human", "flagged": false, "corrected_rating": null }, "chosen_completion": null }, 
#     { "completions": [ 
#         { "text": "That would be n=13.", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 0 }, 
#     { "completions": [ 
#         { "text": "Right, since 8*13+2 is 102.", "rating": -1, "flagged": false }, 
#         { "text": "So there are 13-1+1=13 positive two-digit integers that leave a remainder of 2 when divided by 8.", "rating": -1, "flagged": false }, 
#         { "text": "Because 8*13+2=106.", "rating": 1, "flagged": false }, 
#         { "text": "Because 8*13+2=106 is a three-digit number.", "rating": 1, "flagged": false }, 
#         { "text": "Ok so there are 13-1+1=13 positive two-digit integers that leave a remainder of 2 when divided by 8.", "rating": -1, "flagged": false }, 
#         { "text": "Because 8*13+2=106.", "rating": 1, "flagged": false }, 
#         { "text": "So there are 13-1+1=13 two-digit positive integers that leave a remainder of 2 when divided by 8.\n\n# Answer\n\n13", "rating": -1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 5 }, 
#     { "completions": [ 
#         { "text": "Right. So there are 13-1+1=13 positive two-digit integers that leave a remainder of 2 when divided by 8.\n\n# Answer\n\n13", "rating": -1, "flagged": false }, 
#         { "text": "So there are 13-1+1=13 two-digit integers that leave a remainder of 2 when divided by 8.", "rating": -1, "flagged": false }, 
#         { "text": "So the number of positive two-digit integers of the form 8n+2 is 12.\n\n# Answer\n\n12", "rating": 1, "flagged": false }, 
#         { "text": "Right. So the number of positive two-digit integers that leave a remainder of 2 when divided by 8 is 13.\n\n# Answer\n\n13", "rating": -1, "flagged": false }, 
#         { "text": "Right. But we want to know the number of positive two-digit integers of the form 8n+2.", "rating": 0, "flagged": false }, 
#         { "text": "Right. So the number of positive two-digit integers of the form 8n+2 is 12.", "rating": 0, "flagged": false }, 
#         { "text": "Yes. So the number of positive two-digit integers that leave a remainder of 2 when divided by 8 is 12.\n\n# Answer\n\n12", "rating": 1, "flagged": false } 
#         ], "human_completion": null, "chosen_completion": 2 } 
# ], "total_time": 1099187, "finish_reason": "solution" }

    
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
    "dual_byte_pooling": DualByteLevelProcessor,
    "prm800k": PRM800KProcessor
}



def prepare_data(cfg):
    """
    Split the data, process & tokenize it, and store 
    it as memmap bin files
    """
    # Check if the data is already preprocessed
    dataloader_name = cfg["trainer"]["dataloader"]["name"]
    dataset_names = cfg["trainer"]["dataset"]
    
    # Ensure dataset_names is a list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Create a unique identifier for the combined datasets
    combined_dataset_name = '_'.join(dataset_names)
    
    tokenized_data_folder = os.path.join(
        cfg["general"]["paths"]["data_dir"],
        combined_dataset_name,
        f'{cfg["model"]["tokenizer_type"]}-{cfg["model"]["vocab_size"]}-{dataloader_name}',
    )

    # check if already exists (check len because some datasets use differen filenames
    # (i.e. dual byte level)
    if os.path.exists(tokenized_data_folder) and len(os.listdir(tokenized_data_folder))!=0:
        print("Tokenized data already exists")
        return
    else:
        # create the folder if it doesn't exist   
        if not os.path.exists(tokenized_data_folder):
            os.makedirs(tokenized_data_folder)


    # load embedder
    embedder = build_embedding_model(cfg["model"])

    # load the dataset
    split_dataset = load_data(
        dataset_names=dataset_names,
    )

    processor_object = DATALOADER_PROCESSORS[dataloader_name](
        embedder=embedder
    )

    # wrap in try such that half-complete files can be deleted on error
    try:
        # Get the maximum number of processors
        max_procs = os.cpu_count()
        # cap at 12 to reduce memory usage
        max_procs = min(max_procs, 12) # TODO properly fix this
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


