"""
A simple wrapper around the GPT2 Tokenizer to 
standardize the interface for tokenization.
"""
import tiktoken 

class GPT2Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token
        self.pad_token = 0 # TODO, this is a hack, fix it

    def encode(self, text):
        return self.tokenizer.encode_ordinary(text)
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)