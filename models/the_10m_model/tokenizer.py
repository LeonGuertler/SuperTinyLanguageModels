import os, pickle
from tqdm import tqdm 
import numpy as np 
import tiktoken 

import torch

#from models.utils import load_datasets
from models.utils import load_datasets


TOKENIZERS = {
    "gpt2": lambda: tiktoken.get_encoding("gpt2"),
    "character_basic": lambda classical_tokenizer, config: character_tokenizer(classical_tokenizer,config)
}




class character_bpe_tokenizer:
    def __init__(self, config):
        assert config["arch"]["tokenizer"] in TOKENIZERS

        self.classical_tokenizer = TOKENIZERS["gpt2"]()
        self.character_tokenizer = TOKENIZERS[config["arch"]["tokenizer"]](
            classical_tokenizer=self.classical_tokenizer,
            config=config,
        )
        self.config = config 
        self.dataset_path = os.path.join(
            self.config["paths"]["data_path"],
            self.config["training"]["dataset"],
            self.config["arch"]["tokenizer"]
        )


        self.character_embedding = torch.nn.Embedding(
            num_embeddings=self.config["arch"]["tokenizer_model"]["vocab_size"],
            embedding_dim=self.config["arch"]["tokenizer_model"]["hidden_dim"]
        )

        self.pos_encoding = torch.nn.Embedding(
            num_embeddings=16,
            embedding_dim=self.config["arch"]["tokenizer_model"]["hidden_dim"]
        )

        self.chracter_level_transformer = torch.nn.TransformerEncoderLayer(
            d_model=self.config["arch"]["tokenizer_model"]["hidden_dim"], 
            nhead=self.config["arch"]["tokenizer_model"]["num_heads"], 
            dim_feedforward=self.config["arch"]["tokenizer_model"]["mlp_dim"], 
            dropout=self.config["arch"]["tokenizer_model"]["dropout"],
        )


    def encode_text(self, text, device):
        """
        First encode the text into chracter tokens and bpe token length.
        Then pass the character tokens through the transformer encoder.
        """
        token_ids, token_segments = self.character_tokenizer.encode(text)

        # embed the character tokens
        token_ids = torch.tensor(token_ids, device=device)
        token_segments = torch.tensor(token_segments, device=device)

        # embed the tokens
        token_embeddings = self.character_embedding(token_ids)
        

        # pass the tokens through the transformer




        # TODO
        pass
    
    def decode_tokens(self, token_ids):
        # TODO 
        pass

    def get_batch(self, split="train"):
        # TODO 
        pass


    def prepare_dataset(self):
        """
        Prepare the dataset
        """
        if not os.path.exists(self.dataset_path):
            # check if dataset folder exists
            dataset_folder = os.path.join(
                self.config["paths"]["data_path"],
                self.config["training"]["dataset"]
            )
            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)
                
            # load the dataset 
            dataset = load_datasets(
                self.config["training"]["dataset"]
            )

            
            def process(example):
                ids = self.tokenizer.encode_ordinary(
                    example["text"]
                )
                ids.append(
                    self.tokenizer.eot_token
                )
                return {"ids": ids, "len": len(ids)}
            
            # tokenize dataset
            tokenized = dataset.map(
                process,
                remove_columns=["url", "title", "text"],
                desc="Tokenizing the dataset",
                num_proc=8 # number workers
            )


            # concatenate all the ids in each dataset into one large file for training
            for split, dset in tokenized.items():
                arr_len = np.sum(dset['len'], dtype=np.uint64)
                filename = os.path.join(
                    self.dataset_path, 
                    f'{split}.bin'
                )
                dtype = np.uint16 # possible since enc.max_token_value ==50256 is < 2**16
                arr = np.memmap(
                    filename,
                    dtype=dtype,
                    mode="w+",
                    shape=(arr_len,)
                )
                total_batches = 1024

                idx = 0 
                for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                    batch = dset.shard(
                        num_shards=total_batches,
                        index=batch_idx,
                        contiguous=True
                    ).with_format('numpy')
                    arr_batch = np.concatenate(batch["ids"])

                    # write into mmap
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()



class character_tokenizer:
    def __init__(self, classical_tokenizer, config):
        self.config = config 
        self.classical_tokenizer = classical_tokenizer
        self.dataset_path = os.path.join(
            self.config["paths"]["data_path"],
            self.config["training"]["dataset"],
            self.config["arch"]["tokenizer"]
        )
        self.tokenizer_path = os.path.join(
            self.config["paths"]["data_path"],
            self.config["training"]["dataset"],
            "tokenizer.pickle"
        )
        self._load()

    def _load(self, verbose=True):
        """
        Iterate over dataset and count all unique tokens.
        """
        # check if tokenizer file exists
        if os.path.exists(self.tokenizer_path):
            with open(self.tokenizer_path, "rb") as f:
                self.id_mapping = pickle.load(f)

        else:
            # load the training dataset
            dataset = load_datasets(
                self.config["training"]["dataset"]
            )

            # count token occurences (at the character level)
            token_counts = {}
            for example in tqdm(dataset, desc="Iterating over data samples"):
                for token in example["text"]:
                    if token not in token_counts:
                        token_counts[token] = 0
                    token_counts[token] += 1

            # sort tokens by frequency
            sorted_tokens = sorted(
                token_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # print number of unique tokens 
            # and frequency percentiles
            if verbose:
                print(f"Number of unique tokens: {len(sorted_tokens)}")
                print(f"Most common tokens: {sorted_tokens[:10]}")
                print(f"Least common tokens: {sorted_tokens[-10:]}")

                # plot frequencies as line chart
                import matplotlib.pyplot as plt
                frequency_distribution = {}
                for token, count in sorted_tokens:
                    if count not in frequency_distribution:
                        frequency_distribution[count] = 0
                    frequency_distribution[count] += 1

                # sort
                frequency_distribution = dict(sorted(frequency_distribution.items()))

                #plot
                plt.plot(
                    frequency_distribution.keys(),
                    frequency_distribution.values()
                )
                plt.show()


            # only keep the most frequent config["arch"]["tokenizer_model"]["vocab_size"] tokens
            sorted_tokens = sorted_tokens[
                :self.config["arch"]["tokenizer_model"]["vocab_size"]-self.config["arch"]["tokenizer_model"]["num_special_tokens"]
                ]
            
            sorted_tokens["[unk]"] = 0
            sorted_tokens["[pad]"] = 0
            sorted_tokens["[eos]"] = 0

            # create an ID mapping
            id_mapping = {}
            for i, (token, _) in enumerate(sorted_tokens):
                id_mapping[token] = i

            # create path if not exists
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)

            # store tokenizer
            with open(self.tokenizer_path, "wb") as f:
                pickle.dump(id_mapping, f)

            self.id_mapping = id_mapping


    def encode(self, text):
        """
        Accept text and return a list of token ids and a list of classical token segment lengths.
        """
        token_ids = [] #[self.id_mapping[token] for token in text]
        token_segments = []

        start = 0
        for token in self.classical_tokenizer.encode(text):
            # decode token and store length
            bpe_decoded = self.classical_tokenizer.decode([token])

            bpe_length = len(bpe_decoded)
            token_segments.append((start, start + bpe_length))
            start += bpe_length

            # encode characters
            for char in bpe_decoded:
                if char in self.id_mapping:
                    token_ids.append(self.id_mapping[char])
                else:
                    token_ids.append(self.id_mapping["[unk]"])

        return token_ids, token_segments

    
    def decode(self, token_ids):
        """
        Accept a list of token ids and return the original text.
        """
        return "".join([self.id_mapping[token] for token in token_ids])




