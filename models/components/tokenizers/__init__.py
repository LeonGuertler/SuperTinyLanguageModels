"""
Init file for easier imports of tokenizers.
"""

from models.components.tokenizers.GPT2TokenizerWrapper import GPT2Tokenizer
from models.components.tokenizers.BytePairEncoding import BPETokenizer


def build_tokenizer(tokenizer_type, vocab_size, dataset_name):
    """
    Given the tokenizer config, build it.
    Args:
        tokenizer_cfg: tokenizer_configuration
    Returns:
        tokenizer: tokenizer_instance
    """
    if tokenizer_type == "GPT2":
        return GPT2Tokenizer()
    elif tokenizer_type == "BPE":
        return BPETokenizer(vocab_size=vocab_size, dataset_name=dataset_name)
