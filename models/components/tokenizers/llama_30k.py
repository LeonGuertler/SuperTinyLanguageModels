"""
A simple wrapper around the LLaMA Tokenizer to
standardize the interface for tokenization.
"""

import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from models.components.tokenizers.base_class import Tokenizer

class LLaMATokenizer(Tokenizer):
    """A simple wrapper around the LLaMA Tokenizer."""

    def __init__(self, **_):
        super().__init__()
        # Hardcoded model path from Hugging Face
        model_name_or_path = "decapoda-research/llama-7b-hf"
        model_path = hf_hub_download(repo_id=model_name_or_path, filename="tokenizer.model")
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_path)
        self.eot_token = self.tokenizer.eos_id()
        self.pad_token = self.tokenizer.pad_id()
        self.vocab_size = self.tokenizer.get_piece_size()

    def encode(self, text):
        """Encode a string into tokens."""
        return self.tokenizer.encode(text, out_type=int)

    def encode_batch(self, texts):
        """Encode a list of strings into tokens."""
        return [self.encode(text) for text in texts]

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
        return [self.decode(tokens) for tokens in token_lists]
