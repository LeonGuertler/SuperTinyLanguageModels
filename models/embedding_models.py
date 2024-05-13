"""
A collection of embedding models. A collection model includes 
the tokenizer(s), token embeddings and positional encodings 
(if necessary).
"""
import torch 

from models.components.tokenizers import build_tokenizer
from models.components.positional_encoding import build_positional_encodings



class GenericEmbedder(torch.nn.Module):
    """
    A simple and flexible embedding model
    """
    def __init__(self, model_cfg):
        super().__init__()
        # build the tokenizer
        self.tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["tokenizer_type"],
            vocab_size=model_cfg["vocab_size"],
            dataset_name=model_cfg["dataset_name"]
        )

        # build the token embeddings
        self.token_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["embedding_dim"]
        )

        # build the positional encodings 
        self.positional_encodings = build_positional_encodings(
            model_cfg=model_cfg
        )


    def forward(self, token_ids):
        """
        Takes the token_ids as input 
        and returns the embeddings.
        Args:
            token_ids: torch.tensor(B, S)
        Returns:
            embeddings: torch.tensor(B, S, H)
        """

        # get the token embeddings
        x = self.token_embedder(token_ids)

        # apply the positional encoding 
        x = self.positional_encodings(x)

        return x
    
    def tokenize_input(self, input_string):
        """
        Tokenize an input string.
        """
        return self.tokenizer.encode(input_string)
    
    def inference(self, input_string):
        """
        During inference, tokenize the input string 
        and return the embddings
        Args:
            input_string: str
        Returns:
            embeddings: torch.tensor(B, S, H)
        """
        token_ids = self.tokenize_input(input_string)
        return self.forward(token_ids)