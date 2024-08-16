"""
A simple wrapper around the LLaMA Tokenizer to
standardize the interface for tokenization.
"""


from transformers import AutoTokenizer

class LLaMATokenizer(Tokenizer):
    """A simple wrapper around a LLaMA-based Tokenizer."""

    def __init__(self, **_):
        super().__init__()
        model_name_or_path = "chavinlo/alpaca-native"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
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
