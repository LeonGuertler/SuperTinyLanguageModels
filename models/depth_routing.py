"""
Baseline GPT model (a close copy of NanoGPT)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

# import the layers
from models.layers import LayerNorm, CausalSelfAttention, FFN

from models.embedding import BaselineEmbedder

from models.weight_init import gpt2_weights_init
from models.utils import print_model_stats

class Block(nn.Module):
    """
    A simple abstraction to combine the 
    LayerNorms, SelfAttention and FeedForward layers
    """
    def __init__(self, hidden_dim, ffn_dim, bias, num_heads, dropout):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = FFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            dropout=dropout,
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

def mask_tokens(token_certainty_mask, attention_mask):
    """
    use the token_certainty_mask to alter the attention_mask
    specically, blocks updates to tokens that are not certain
    """
    b, s = token_certainty_mask.size()
    assert attention_mask.size() == (b, s, s)
    token_certainty_mask = token_certainty_mask.unsqueeze(-1)
    return attention_mask * token_certainty_mask

class DepthRouter(BaseGPT):

    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg["vocab_size"] is not None
        assert cfg["context_window"] is not None

        self.cfg = cfg

        # construct the actual model
        self.embedder =  BaselineEmbedder(
            hidden_dim=cfg["hidden_dim"],
            context_window=cfg["context_window"],
            vocab_size=cfg["vocab_size"],
        )
        assert cfg["depth"] > 2, "Depth must be greater than 2"
        self.uncertainty_projections = nn.ModuleDict(
            dict(
                h=nn.ModuleList(
                    [nn.Linear(cfg["hidden_dim"], 1) for _ in range(cfg["depth"])]
                )
            
            )
        )
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(cfg["dropout"]),
                h=nn.ModuleList(
                    [Block(
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

        # check if vocab size is the same as the number of tokens
        #assert (
        #    self.embedder.tokenizer.max_token_value == cfg["vocab_size"]
        #), f"Vocab size ({cfg['vocab_size']}) must be the same as the number of tokens in the tokenizer ({self.embedder.tokenizer.max_token_value})"

        # share the weights between the token embeddings and the final logit layer
        #self.embedder.embedding.weight = (
        #    self.lm_head.linear.weight
        #) # https://paperswithcode.com/method/weight-tying


        # init all weights
        self.apply(lambda module: gpt2_weights_init(module, self.cfg["depth"]))

        # report number of parameters
        print_model_stats(self)
