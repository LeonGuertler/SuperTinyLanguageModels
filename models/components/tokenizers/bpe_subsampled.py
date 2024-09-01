import os
from typing import List
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import json 
from transformers import AutoTokenizer
from tokenizers import Tokenizer 


from models.components.tokenizers import utils
from models.components.tokenizers.base_class import Tokenizer as BaseTokenizer
from trainers.utils import load_data

class BPESubsampledTokenizer(BaseTokenizer):
    """Tokenizer for Byte Pair Encoding using Hugging Face tokenizers library."""

    def __init__(self, vocab_size: int):
        """
        Load and subsample the Llama-3.1 tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B", use_fast=True)
        assert self.tokenizer.is_fast, "Tokenizer is not fast"
        tokenizer_json = json.loads(self.tokenizer._tokenizer.to_str())
        vocab = tokenizer_json["model"]["vocab"]

        # Prune the vocabulary
        new_vocab = {token: i for i, (token, _) in enumerate(vocab.items()) if i < vocab_size-3}
        merges = tokenizer_json["model"]["merges"]
        new_merges = []
        for i in range(len(merges)):
            a, b = merges[i].split()
            new_token = " ".join([a, b])
            if a in new_vocab and b in new_vocab and new_token in new_vocab:
                new_merges.append(merges[i])

        # add the special tokens (eot, pad, unk)
        new_vocab["<|pad|>"] = vocab_size - 3
        new_vocab["<|endoftext|>"] = vocab_size - 2
        new_vocab["<|unk|>"] = vocab_size - 1

        tokenizer_json["model"]["merges"] = new_merges 
        tokenizer_json["model"]["vocab"] = new_vocab
        self.tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

        self.vocab_size = vocab_size
        self.eot_token = vocab_size - 2
        self.pad_token = vocab_size - 3
        self.unk_token = vocab_size - 1



    def encode(self, text: str) -> List[int]:
        """
        Encode the text into BPE tokens.
        
        Args:
            text (str): The input text to encode.
        
        Returns:
            List[int]: The list of token ids.
        """
        return self.tokenizer(text)


    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts into BPE tokens.
        
        Args:
            texts (List[str]): The list of input texts to encode.
        
        Returns:
            List[List[int]]: The list of token id lists.
        """
        return [self.encode(text) for text in texts]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode the BPE tokens back into text.
        
        Args:
            tokens (List[int]): The list of token ids to decode.
        
        Returns:
            str: The decoded text.
        """
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode a batch of BPE token lists back into text.
        
        Args:
            token_lists (List[List[int]]): The list of token id lists to decode.
        
        Returns:
            List[str]: The list of decoded texts.
        """
        if torch.is_tensor(token_lists):
            token_lists = token_lists.tolist()
        return [self.decode(tokens) for tokens in token_lists]

    def _train_tokenizer(self, verbose: bool = True):
        """
        No need to train
        """
        pass

    def _save(self):
        """
        Save the tokenizer as a .json file.
        """
        tokenizer_folder, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="bpe",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
        )
        if not os.path.exists(tokenizer_folder):
            os.makedirs(tokenizer_folder)

        self.tokenizer.save(tokenizer_path)

    def _load(self):
        """
        No need to load
        """
        pass