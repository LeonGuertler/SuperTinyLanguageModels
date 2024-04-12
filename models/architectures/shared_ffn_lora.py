"""
Share the FFN weights across all the layers,
but add individual layer loras.
"""



import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

# import the layers
from models.layers import (
    LayerNorm, 
    CausalSelfAttention,
    LoraLinear,
    NextTokenHead
)

from models.embedding import BaselineEmbedder

from models.weight_init import gpt2_weights_init
from models.utils import print_model_stats



class LoraFNN(nn.Module):
    """
    A simple Feed Forward Network block with lora channel
    """
    def __init__(self, hidden_dim, ffn_dim, bias=False, dropout=0.0, rank=32, lora_weighting=1):
        super().__init__()
        self.c_fc = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=bias,
        )

        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            ffn_dim,
            hidden_dim,
            bias=bias,
        )
        self.dropout = nn.Dropout(
            dropout
        )

        self.lora_weighting = lora_weighting
        self.lora_lin_down = nn.Linear(
            hidden_dim,
            rank
        )
        self.lora_lin_up = nn.Linear(
            rank,
            hidden_dim
        )

    def forward(self, x):
        """
        Forward pass
        """
        lx = self.lora_lin_down(x)
        x = self.c_fc(x)

        x = self.gelu(x)
        lx = self.gelu(lx)

        lx = self.lora_lin_up(lx)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x + lx * self.lora_weighting

class LoraBlock(nn.Module):
    """
    A simple abstraction to combine the 
    LayerNorms, SelfAttention and FeedForward layers
    """
    def __init__(self, hidden_dim, ffn_dim, bias, num_heads, dropout, rank=32, lora_weighting=1):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = LoraFNN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            dropout=dropout,
            rank=rank,
            lora_weighting=lora_weighting
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward 
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x



class SharedFNNLora(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg["vocab_size"] is not None
        assert cfg["context_window"] is not None

        self.cfg = cfg

        # construct the actual model
        self.embedder =  BaselineEmbedder(
            hidden_dim=cfg["hidden_dim"],
            context_window=cfg["context_window"],
            vocab_size=cfg["vocab_size"],
            tokenizer_name=cfg["tokenizer"]
        )
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(cfg["dropout"]),
                h=nn.ModuleList(
                    [LoraBlock(
                        hidden_dim=cfg["hidden_dim"], 
                        ffn_dim=cfg["ffn_dim"], 
                        bias=cfg["bias"], 
                        num_heads=cfg["num_heads"], 
                        dropout=cfg["dropout"],
                    ) for _ in range(cfg["depth"])]
                )
            )
        )

        self.lm_head = NextTokenHead(
            hidden_dim=cfg["hidden_dim"],
            vocab_size=cfg["vocab_size"],
        )

        # share ffn weights
        for block in self.transformer.h:
            block.mlp.c_fc.weight = self.transformer.h[0].mlp.c_fc.weight
            block.mlp.c_fc.bias = self.transformer.h[0].mlp.c_fc.bias
            block.mlp.c_proj.weight = self.transformer.h[0].mlp.c_proj.weight
            block.mlp.c_proj.bias = self.transformer.h[0].mlp.c_proj.bias

        # check if vocab size is the same as the number of tokens
        #assert (
        #    self.embedder.tokenizer.max_token_value == cfg["vocab_size"]
        #), f"Vocab size ({cfg['vocab_size']}) must be the same as the number of tokens in the tokenizer ({self.embedder.tokenizer.max_token_value})"

        # share the weights between the token embeddings and the final logit layer
        self.embedder.embedding.weight = (
            self.lm_head.linear.weight
        ) # https://paperswithcode.com/method/weight-tying


        # init all weights
        self.apply(lambda module: gpt2_weights_init(module, self.cfg["depth"]))

        # report number of parameters
        print_model_stats(self)



    def feature_extraction(self, token_ids):
        """
        Use the model to get the text features.
        """
        b, s = token_ids.size()

        # check that the sequence length is not longer than the context window
        assert (
            s <= self.cfg["context_window"]
        ), f"Cannot forward sequence of length {s}, block size is only {self.cfg['context_window']}"

        # embed and pos-encode the tokens
        x = self.embedder.embed_tokens(token_ids)

        # forward through the GPT transformer
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)

        return x

    def forward(self, token_ids):
        """
        The default forward pass is used for training and accepts the 
        token_ids as input. When the model is in eval mode, only the 
        last token is passed into the NextTokenHead.
        """
        # extract the features
        x = self.feature_extraction(token_ids)        

        # forward the entire sequence through the lm_head
        logits = self.lm_head(x)
        return logits 

        
    def inference(self, text_string):
        """
        Similar to the forward pass, but takes in a string 
        (or batch of strings) and only return the logits 
        for the next token.
        Args:
            text_string: a string or list of strings
        Returns:
            logits for the next token
        """
        # fully encode the text string (or batch of text string)
        x, attention_mask = self.embedder(
            text_string,
            pad_truncate=True)
        
        # extract the features
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)

        # forward only the last token through the lm_head
        logits = self.lm_head(x[:, -1, :])
        return logits

        

