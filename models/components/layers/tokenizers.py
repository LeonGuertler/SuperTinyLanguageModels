""" A collection of different tokenizers. """
from typing import List

import torch
import tiktoken
from transformers import AutoTokenizer

# local imports
from trainers.data_utils import load_data
from models.components.layers import utils

# text processing imports
import re
import string

# Custom BPE tokenizer imports
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import PreTokenizer, WhitespaceSplit, Sequence, Split, ByteLevel
from tokenizers.normalizers import NFD, StripAccents, Replace, Sequence as NormalizerSequence
from tokenizers import decoders


import os
import re
import string
import json
from typing import List, Dict
import torch  # Assuming torch is used elsewhere in your project


class TokenizerClass:
    """Base class for tokenizers, defines the interface for tokenizers."""

    def __init__(self, **_):
        self.eot_token = 0
        self.pad_token = 0
        self.vocab_size = ...

    def encode(self, text):
        """Encode a text into tokens."""
        raise NotImplementedError

    def encode_batch(self, texts):
        """Encode a batch of texts into tokens.

        Default implementation is to loop over the texts"""
        for text in texts:
            yield self.encode(text)

    def pad_batch(self, token_lists, direction="right"):
        """Pad a list of token lists to the same length,
        and return the padded tensor, and mask tensor.
        
        Direction can be 'right' or 'left' to specify the padding direction.
        """
        max_len = max(len(tokens) for tokens in token_lists)
        padded_tokens = []
        mask = []
        for tokens in token_lists:
            if direction == "right":
                padded_tokens.append(tokens + [self.pad_token] * (max_len - len(tokens)))
                mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
            elif direction == "left":
                padded_tokens.append([self.pad_token] * (max_len - len(tokens)) + tokens)
                mask.append([0] * (max_len - len(tokens)) + [1] * len(tokens))
        return torch.tensor(padded_tokens), torch.tensor(mask)

    def decode(self, tokens):
        """Decode a list of tokens into a string."""
        raise NotImplementedError

    def decode_batch(self, token_lists):
        """Decode a list of token lists into a list of strings.

        Default implementation is to loop over the token lists."""
        for tokens in token_lists:
            yield self.decode(tokens)


class HuggingfaceTokenizer(TokenizerClass):
    """A simple wrapper around a Huggingface Tokenizer."""

    def __init__(self, tokenizer_path):
        super().__init__()
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.eot_token = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token_id
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text):
        """Encode a string into tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(self, texts):
        """Encode a list of strings into tokens."""
        return [self.encode(text) for text in texts]

    def decode(self, tokens):
        """Decode a list of tokens into a string."""
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def decode_batch(self, token_lists):
        """Decode a list of token lists into a list of strings."""
        if torch.is_tensor(token_lists):
            token_lists = token_lists.tolist()
        return [self.decode(tokens) for tokens in token_lists]

class TiktokenTokenizer(TokenizerClass):
    """A simple wrapper around the GPT2 Tokenizer."""

    def __init__(self, tokenizer_name: str):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.eot_token = self.tokenizer.eot_token
        self.pad_token = self.tokenizer.eot_token
        self.vocab_size = self.tokenizer.max_token_value + 1

    def encode(self, text):
        """Encode a string into tokens."""
        return self.tokenizer.encode_ordinary(text)

    def encode_batch(self, texts):
        """Encode a list of strings into tokens."""
        return self.tokenizer.encode_ordinary_batch(texts)

    def decode(self, tokens):
        """Decode a list of tokens into a string."""
        # check if the tokens are a tensor
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def decode_batch(self, token_lists):
        """Decode a list of token lists into a list of strings."""
        if torch.is_tensor(token_lists):
            token_lists = token_lists.tolist()
        return self.tokenizer.decode_batch(token_lists)



class BPETokenizer_old(TokenizerClass):
    def __init__(self, vocab_size: int, dataset_name: str, simplify: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dataset_name = dataset_name
        self.simplify = simplify 

        if not utils.check_if_tokenizer_exists(
            tokenizer_type="bpe", 
            vocab_size=vocab_size, 
            dataset_name=dataset_name,
            simplify=simplify
        ):
            self._train_tokenizer()
            self._save()
        else:
            self._load()
        
        self.pad_token = self.tokenizer.token_to_id("[PAD]")
        self.eot_token = self.tokenizer.token_to_id("[EOT]")
        self.unk_token = self.tokenizer.token_to_id("[UNK]") 

    def _train_tokenizer(self, verbose: bool = True):
        raw_datasets = load_data(dataset_name=self.dataset_name)

        # Pattern string without compiling
        non_english_char_pattern = r'[^a-zA-Z0-9\s' + re.escape(string.punctuation) + r']'

        # Define special tokens
        special_tokens = ["[PAD]", "[EOT]", "[UNK]"]

        # Define initial alphabet, include digits to ensure they are treated as individual tokens
        initial_alphabet = list(string.ascii_letters + string.digits + string.punctuation + ' \n\t')
        
        
        # Initialize a new tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]", dropout=0.1))
        
        # Set the decoder to ByteLevel
        self.tokenizer.decoder = decoders.ByteLevel() 
        
        # Initialize the trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=5,
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
            show_progress=verbose
        )

        if self.simplify:
            # Set the normalizer to remove non-English characters
            self.tokenizer.normalizer = NormalizerSequence([
                NFD(),  # Decompose unicode characters
                StripAccents(),  # Remove accents
                Replace(non_english_char_pattern, ""),  # Use pattern string directly
        ])

            # Custom pre-tokenizer to split numbers into individual digits
            self.tokenizer.pre_tokenizer = Sequence([
                WhitespaceSplit(),  # Split on whitespace
                # Split digits and isolate them
                Split(r'\d', behavior='isolated'),  # Each digit is a separate token
                ByteLevel()  # Byte-level encoding
            ])
        
        # Prepare the training data with filtering
        def batch_iterator():
            batch_size = 1000
            for i in range(0, len(raw_datasets["train"]), batch_size):
                batch_texts = raw_datasets["train"][i:i+batch_size]["text"]
                # Filter and clean texts
                cleaned_texts = []
                for text in batch_texts:
                    text = text.strip()
                    # Remove non-English characters
                    cleaned_text = re.sub(non_english_char_pattern, '', text)
                    if cleaned_text:
                        cleaned_texts.append(cleaned_text)
                if cleaned_texts:
                    yield cleaned_texts
        
        # Train the tokenizer
        self.tokenizer.train_from_iterator(
            batch_iterator(), 
            trainer=trainer
        )

        # print 
        if verbose:
            print(f"Trained a BPE tokenizer with {self.vocab_size} tokens on the {self.dataset_name} dataset.")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode(self, tokens: List[int]) -> str:
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        if torch.is_tensor(token_lists):
            token_lists = token_lists.tolist()
        return [self.decode(tokens) for tokens in token_lists]

    def _save(self):
        _, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="bpe",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
            simplify=self.simplify
        )
        self.tokenizer.save(str(tokenizer_path)) 

    def _load(self):
        _, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="bpe",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
            simplify=self.simplify
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab = self.tokenizer.get_vocab()

        # print vocab size
        print(f"Loaded a BPE tokenizer with {len(self.vocab)} tokens.")

class BPETokenizer:
    """
    A byte-level tokenizer with a fixed vocabulary of 256 bytes plus three special tokens:
    [PAD], [EOT], [UNK], totaling 259 tokens.
    """
    
    # Define special tokens and their IDs
    SPECIAL_TOKENS = {
        "[PAD]": 256,
        "[EOT]": 257,
        "[UNK]": 258
    }
    
    def __init__(self, vocab_size: int = 259, dataset_name: str = "default_dataset", simplify: bool = True):
        """
        Initialize the BPETokenizer.

        Args:
            vocab_size (int): Size of the vocabulary. Fixed to 259.
            dataset_name (str): Name of the dataset to be used.
            simplify (bool): Whether to simplify the text by removing non-English characters.
        """
        self.vocab_size = vocab_size  # Fixed to 259
        self.dataset_name = dataset_name
        self.simplify = simplify 
        
        # Initialize the vocabulary
        self.vocab = self._build_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}  # For decoding
        
        # Set special token IDs
        self.pad_token = self.vocab["[PAD]"]
        self.eot_token = self.vocab["[EOT]"]
        self.unk_token = self.vocab["[UNK]"]
        
        # Check if tokenizer exists; if not, save the current vocab
        if not utils.check_if_tokenizer_exists(
            tokenizer_type="byte", 
            vocab_size=self.vocab_size, 
            dataset_name=self.dataset_name,
            simplify=self.simplify
        ):
            self._save()
        else:
            self._load()
    
    def _build_vocab(self) -> Dict[str, int]:
        """
        Build the vocabulary mapping bytes and special tokens to unique IDs.

        Returns:
            Dict[str, int]: The vocabulary mapping.
        """
        vocab = {chr(i): i for i in range(256)}  # Byte tokens: 0-255
        vocab.update(self.SPECIAL_TOKENS)        # Special tokens: 256-258
        return vocab
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a single string into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: The list of token IDs.
        """
        if self.simplify:
            text = self._simplify_text(text)
        
        byte_sequence = text.encode('utf-8', errors='replace')  # Encode to bytes
        token_ids = []
        
        for byte in byte_sequence:
            char = chr(byte)
            token_id = self.vocab.get(char, self.unk_token)
            token_ids.append(token_id)
        
        token_ids.append(self.eot_token)  # Append end-of-text token
        assert all([type(i)==int for i in token_ids]), token_ids
        assert len(token_ids)>0, token_ids
        return token_ids
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of strings into lists of token IDs.

        Args:
            texts (List[str]): The list of input texts to encode.

        Returns:
            List[List[int]]: The list of token ID lists.
        """
        return [self.encode(text) for text in texts]
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            tokens (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        # Remove special tokens if present
        tokens = [token for token in tokens if token not in self.SPECIAL_TOKENS.values()]
        
        byte_sequence = bytes([self.inv_vocab.get(token, ord('?')) for token in tokens])  # Replace unknown tokens with '?'
        try:
            text = byte_sequence.decode('utf-8', errors='replace')  # Decode bytes to string
        except UnicodeDecodeError:
            text = byte_sequence.decode('utf-8', errors='replace')
        return text
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode a batch of token ID lists back into strings.

        Args:
            token_lists (List[List[int]]): The list of token ID lists to decode.

        Returns:
            List[str]: The list of decoded strings.
        """
        return [self.decode(tokens) for tokens in token_lists]
    
    def _simplify_text(self, text: str) -> str:
        """
        Simplify the text by removing non-English characters based on a predefined pattern.

        Args:
            text (str): The input text to simplify.

        Returns:
            str: The simplified text.
        """
        # Define pattern to remove non-English characters
        non_english_char_pattern = r'[^a-zA-Z0-9\s' + re.escape(string.punctuation) + r']'
        cleaned_text = re.sub(non_english_char_pattern, '', text)
        return cleaned_text
    
    def _save(self):
        """
        Save the tokenizer's vocabulary to a file for later use.
        """
        _, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="byte",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
            simplify=self.simplify
        )
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        
        print(f"Saved byte-level tokenizer to {tokenizer_path}.")
    
    def _load(self):
        """
        Load the tokenizer's vocabulary from a file.
        """
        _, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="byte",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
            simplify=self.simplify
        )
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}.")
        
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}  # Update inverse vocab
        
        # Update special tokens
        self.pad_token = self.vocab.get("[PAD]", 256)
        self.eot_token = self.vocab.get("[EOT]", 257)
        self.unk_token = self.vocab.get("[UNK]", 258)
        
        print(f"Loaded byte-level tokenizer from {tokenizer_path} with {len(self.vocab)} tokens.")



TOKENIZER_DICT = {
    # a number of standard tiktoken tokenizers
    "o200k_base": lambda vocab_size, dataset_name, simplify: TiktokenTokenizer(tokenizer_name="o200k_base"),
    "cl100k_base": lambda vocab_size, dataset_name, simplify: TiktokenTokenizer(tokenizer_name="cl100k_base"),
    "p50k_base": lambda vocab_size, dataset_name, simplify: TiktokenTokenizer(tokenizer_name="p50k_base"),
    "gpt2": lambda vocab_size, dataset_name, simplify: TiktokenTokenizer(tokenizer_name="gpt2"),

    # a number of standard huggingface tokenizers
    "llama_32k": lambda vocab_size, dataset_name, simplify: HuggingfaceTokenizer(tokenizer_path="chavinlo/alpaca-native"),
    "opt_50k": lambda vocab_size, dataset_name, simplify: HuggingfaceTokenizer(tokenizer_path="facebook/opt-1.3b"),
    "mistral_32k": lambda vocab_size, dataset_name, simplify: HuggingfaceTokenizer(tokenizer_path="mistralai/Mistral-7B-v0.1"),

    # a custom BPE tokenizer (using the HF implementation)
    "bpe": lambda vocab_size, dataset_name, simplify: BPETokenizer(
        vocab_size=vocab_size, dataset_name=dataset_name, simplify=simplify
    ),
}


def build_tokenizer(
        tokenizer_type, 
        vocab_size, 
        dataset_name,
        simplify,
    ) -> TokenizerClass:
    """
    Build the tokenizer.
    """
    assert tokenizer_type in TOKENIZER_DICT, \
        f"Tokenizer type {tokenizer_type} not found. The available tokenizers are: {list(TOKENIZER_DICT.keys())}"
    return TOKENIZER_DICT[tokenizer_type](
        vocab_size=vocab_size, dataset_name=dataset_name, simplify=simplify
    )
