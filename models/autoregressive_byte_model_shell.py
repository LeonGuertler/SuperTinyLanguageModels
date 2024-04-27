"""
The Model Shell holds the tokenizer, core-model and model head.
"""

import torch 
import torch.nn as nn

from models.components.tokenizers import build_tokenizer
from models.components.LMHeads import (
    NextTokenHead
)

from models.utils import (
    print_model_stats
)

from models.components.layers import BidirectionalTransformerBlock



class AutoregressiveByteModelShell(nn.Module):
    """
    A model shell for byte-level learning.
    """
    def __init__(
            self,
            cfg,
            core_model,
        ):
        super().__init__()

        # move to class
        self.cfg = cfg
        self.core_model = core_model

        # build the tokenizer 
        self.byte_tokenizer = build_tokenizer(
            tokenizer_type=self.cfg["model_shell"]["pooling_tokenizer"],
            vocab_size=self.cfg["model_shell"]["pooling_vocab_size"],
            dataset_name=self.cfg["model_shell"]["tokenizer_dataset_name"],
        )

        self.pooling_tokenizer = build_tokenizer(
            tokenizer_type=self.cfg["model_shell"]["tokenizer"],
            vocab_size=self.cfg["model_shell"]["vocab_size"],
            dataset_name=self.cfg["model_shell"]["tokenizer_dataset_name"],
        )

        # build the embedder 
        self.token_embedder = nn.Embedding(
            num_embeddings=self.cfg["model_shell"]["vocab_size"],
            embedding_dim=self.cfg["model_shell"]["embedding_dim"],
        )

        self.byte_token_processor = ByteLevelProcessor(
            embedding_dim=self.cfg["model_shell"]["embedding_dim"],
            hidden_dim=self.cfg["core_model"]["hidden_dim"],
            pooling_tokenizer=self.pooling_tokenizer,
            byte_tokenizer=self.byte_tokenizer,
            token_embedder=self.token_embedder
        )

        # build the language model head
        self.lm_head = NextTokenHead(
            hidden_dim=self.cfg["core_model"]["hidden_dim"],
            vocab_size=self.cfg["model_shell"]["vocab_size"],
        )

        # share the weights between the token embeddings and the final logit layer
        #self.token_embedder.weight = (
        #    self.lm_head.linear.weight
        #) # https://paperswithcode.com/method/weight-tying


        # report number of parameters
        print_model_stats(self)


    def forward(self, token_ids):
        """
        The default forward pass is used for training and accepts the 
        token_ids as input. When the model is in eval mode, only the 
        last token is passed into the NextTokenHead.
        """

        #b, s = token_ids.size()

        # check that the sequence length is not longer than the context window
        #assert (
        #    s <= self.cfg["model_shell"]["context_window"]
        #), f"Cannot forward sequence of length {s}, block size is only {self.cfg['model_shell']['context_window']}"


        # embed token_ids
        #x = self.token_embedder(token_ids)

        # process to sub-word tokens
        x = self.byte_token_processor(token_ids)

        # forward through the core model
        x_return = self.core_model(x)
        if isinstance(x, tuple):
            x, loss = x_return
        else:
            x, loss = x_return, None

        # get logits
        logits = self.lm_head(x)

        return logits, loss
        
    def inference(self, token_ids):
        """
        Similar to the forward pass, but takes in a string 
        (or batch of strings) and only return the logits 
        for the next token.
        Args:
            text_string: a string or list of strings
        Returns:
            logits for the next token
        """

        b, s = token_ids.size()

        # check that the sequence length is not longer than the context window
        assert (
            s <= self.cfg["model_shell"]["context_window"]
        ), f"Cannot forward sequence of length {s}, block size is only {self.cfg['model_shell']['context_window']}"


        # embed token_ids
        #x = self.token_embedder(token_ids)

        # process to sub-word tokens
        x = self.byte_token_processor(token_ids)

        # forward through the core model
        x_return = self.core_model(x)
        if isinstance(x, tuple):
            x, loss = x_return
        else:
            x, loss = x_return, None

        # get logits
        logits = self.lm_head(x)

        return logits[:, -1, :]





class ByteLevelProcessor(nn.Module):
    """
    Takes byte level encodings, processes them via 
    two local-attention transformer blocks and pools 
    the resultant tokens based on gpt-2 tokenizer 
    boundaries.
    Inputs are batches of lists of token blocks 
    in the gpt2 tokenizer boundaries.
    """
    def __init__(self, embedding_dim, hidden_dim, pooling_tokenizer, byte_tokenizer, token_embedder):
        super().__init__() 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pooling_tokenizer = pooling_tokenizer
        self.byte_tokenizer = byte_tokenizer
        self.token_embedder = token_embedder
        self.transformer = nn.ModuleList(
            [
                BidirectionalTransformerBlock(
                    hidden_dim=embedding_dim,
                    ffn_dim=embedding_dim*4,
                    ffn_activation="gelu",
                    bias=False,
                    num_heads=8,
                    dropout=0.0
                ),
                BidirectionalTransformerBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=hidden_dim*4,
                    ffn_activation="gelu",
                    bias=False,
                    num_heads=8,
                    dropout=0.0
                ),
            ]
        )
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)


    def forward(self, x):
        """
        A batch of sequences of byte tokens.
        First flatten across first dim, pass through transformer, pool and reshape
        """
        input(x.size())

        B, S, S_char = x.size()

        x = x.view(B*S, S_char)

        # embed 
        x = self.token_embedder(x)
        x = self.transformer[0](x)
        x = self.up_proj(x)
        x = self.transformer[1](x)
        x = x.mean(dim=-2)
        x = x.view(B, S, -1)
        input(x.size())
        return x
