"""
A script for building the various tokenizers.
"""

from models.components.tokenizers.base_class import Tokenizer
from models.components.tokenizers.bpe import BPETokenizer
from models.components.tokenizers.gpt2 import GPT2Tokenizer

TOKENIZER_DICT = {
    "gpt2": lambda vocab_size, dataset_name: GPT2Tokenizer(),
    "bpe": lambda vocab_size, dataset_name: BPETokenizer(
        vocab_size=vocab_size, dataset_name=dataset_name
    ),
}


def build_tokenizer(tokenizer_type, vocab_size, dataset_name) -> Tokenizer:
    """
    Build the tokenizer.
    """
    return TOKENIZER_DICT[tokenizer_type](
        vocab_size=vocab_size, dataset_name=dataset_name
    )
