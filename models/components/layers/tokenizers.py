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
from tokenizers.pre_tokenizers import PreTokenizer, WhitespaceSplit, Sequence, Split
from tokenizers.normalizers import NFD, StripAccents, Replace, Sequence as NormalizerSequence


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



class BPETokenizer(TokenizerClass):
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
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab = self.tokenizer.get_vocab()

        # print vocab size
        print(f"Loaded a BPE tokenizer with {len(self.vocab)} tokens.")




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
