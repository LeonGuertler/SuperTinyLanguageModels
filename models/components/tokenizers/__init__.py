"""
Init file for easier imports of tokenizers.
"""

from models.components.tokenizers.GPT2TokenizerWrapper import GPT2Tokenizer
from models.components.tokenizers.BytePairEncoding import BPETokenizer


TOKENIZER_DICT = {
    "gpt2": lambda vocab_size, dataset_name: GPT2Tokenizer(),
    "bpe": lambda vocab_size, dataset_name: BPETokenizer(vocab_size, dataset_name),
}
def build_tokenizer(tokenizer_type, vocab_size, dataset_name):
    """
    Given the tokenizer config, build it.
    Args:
        tokenizer_cfg: tokenizer_configuration
    Returns:
        tokenizer: tokenizer_instance
    """
    return TOKENIZER_DICT[tokenizer_type](
        vocab_size=vocab_size,
        dataset_name=dataset_name
    )