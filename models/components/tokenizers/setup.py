"""
A script for building the various tokenizers.
"""

from models.components.tokenizers.base_class import Tokenizer
from models.components.tokenizers.bpe import BPETokenizer
from models.components.tokenizers.gpt2 import GPT2Tokenizer
from models.components.tokenizers.ck100k import CL100KTokenizer

TOKENIZER_DICT = {
    "gpt2": lambda vocab_size, dataset_name: GPT2Tokenizer(),
    "bpe": lambda vocab_size, dataset_name: BPETokenizer(
        vocab_size=vocab_size, dataset_name=dataset_name
    ),
    "cl100k": lambda vocab_size, dataset_name: CL100KTokenizer(),
}


def build_tokenizer(tokenizer_type, vocab_size, dataset_name) -> Tokenizer:
    """
    Build the tokenizer.
    """
    return TOKENIZER_DICT[tokenizer_type](
        vocab_size=vocab_size, dataset_name=dataset_name
    )
