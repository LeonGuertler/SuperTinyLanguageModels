class DualBytePooling(DatasetInterface):
    """
    Dataset for both byte-level and higher token level tokens simultaneously
    """
    def __init__(self, split, cfg):
        self.loading_shape = None
        # overwrite datapath
        data_folder = os.path.join(
            cfg["general"]["paths"]["data_dir"],
            cfg["trainer"]["dataset"],
            f'{cfg["model"]["embedder"]["tokenizer_type"]}-{cfg["model"]["vocab_size"]}-{cfg["trainer"]["dataloader"]["name"]}',
        )
        self.data_path_byte = os.path.join(data_folder, f"{split}_byte.bin")
        self.data_path_token = os.path.join(data_folder, f"{split}_token.bin")
        super().__init__(split, cfg)

        # force parent init
        self._load_data()

    def _load_data(self):
        """
        Get both the byte-level and the token level data
        """
        if self.loading_shape is None:
            data = np.memmap(
                self.data_path_byte,
                dtype=np.uint16,
                mode="r",
            )
            self.loading_shape = (len(data)// self.cfg["model"]["embedder"]["byte_context_window"], self.cfg["model"]["embedder"]["byte_context_window"])
            data = None
        self.data_byte = np.memmap(
            self.data_path_byte,
            dtype=np.uint16,
            mode="r",
            shape=self.loading_shape,
        )
        self.data = np.memmap(
            self.data_path_token,
            dtype=np.uint16,
            mode="r",
        )
    
    def __getitem__(self, idx):
        """
        Get a batch of data from both the byte and higher token level
        """
        # get byte level batch
        x_byte = torch.from_numpy((self.data_byte[idx: idx + self.context_window]).astype(np.int64))
        #y_byte = torch.from_numpy((self.data_byte[idx + 1: idx + 1 + self.context_window]).astype(np.int64))

        # get token level batch
        #x_token = torch.from_numpy((self.data_token[idx: idx + self.context_window]).astype(np.int64))
        y_token = torch.from_numpy((self.data[idx + 1: idx + 1 + self.context_window]).astype(np.int64))
        return x_byte, y_token  


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



    def tokenize_input(self, input_string: str, truncate=False, add_eot=True, return_high_level=False):
        """Tokenize an input string.

        In this case we actually want to pre-tokenize using the pooling tokenizer,
        the byte tokenizer is then used in the forward pass. Its a bit complicated...
        """
        pooling_ids = self.pooling_tokenizer.encode(input_string)
        if add_eot:
            pooling_ids += [self.pooling_tokenizer.eot_token]
        if truncate:
            pooling_ids = self.truncate([pooling_ids])[0]
        tokens = [
            self.byte_tokenizer.encode(self.pooling_tokenizer.decode([pool_id]))
            for pool_id in pooling_ids
        ]
        # truncate bytes
        tokens = [
            token_seq[: self.model_cfg["byte_context_window"]] for token_seq in tokens
        ]
        # pad bytes
        tokens = [
            token_seq
            + [self.byte_tokenizer.pad_token]
            * (self.model_cfg["byte_context_window"] - len(token_seq))
            for token_seq in tokens
        ]
        if not return_high_level:
            return tokens
        else:
            return tokens, pooling_ids