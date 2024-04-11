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
        time_dict = {}
        start_block = torch.cuda.Event(enable_timing=True)
        end_block = torch.cuda.Event(enable_timing=True)
        start_block.record()

        x1 = self.ln_1(x)
        end_block.record()
        torch.cuda.synchronize()
        time_dict = {"ln1": start_block.elapsed_time(end_block)}

        start_block.record()
        x2 = self.attn(x1, attention_mask)
        end_block.record()
        torch.cuda.synchronize()
        time_dict["attn"] = start_block.elapsed_time(end_block)

        start_block.record()
        x = x + x2
        end_block.record()
        torch.cuda.synchronize()
        time_dict["attn_residual"] = start_block.elapsed_time(end_block)

        start_block.record()
        x1 = self.ln_2(x)
        end_block.record()
        torch.cuda.synchronize()
        time_dict["ln2"] = start_block.elapsed_time(end_block)

        start_block.record()
        x2 = self.mlp(x1)
        end_block.record()
        torch.cuda.synchronize()
        time_dict["mlp"] = start_block.elapsed_time(end_block)

        start_block.record()
        x = x + x2
        end_block.record()
        torch.cuda.synchronize()
        time_dict["mlp_residual"] = start_block.elapsed_time(end_block)


        return x
    
class NextTokenHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.ln = LayerNorm(hidden_dim, bias=True)
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.ln(x)
        logits = self.linear(x)
        return logits


class BaseGPT(nn.Module):

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
        time_dict = {}
        # extract the features
        b, s = token_ids.size()

        # check that the sequence length is not longer than the context window
        assert (
            s <= self.cfg["context_window"]
        ), f"Cannot forward sequence of length {s}, block size is only {self.cfg['context_window']}"

        # embed and pos-encode the tokens
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        x = self.embedder.embed_tokens(token_ids)
        end.record()
        torch.cuda.synchronize()
        time_dict["embed_tokens"] = start.elapsed_time(end)

        # forward through the GPT transformer
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            start.record()
            x, time_dict_mini = block(x)
            end.record()
            torch.cuda.synchronize()
            time_dict[f"block_{i}_full"] = start.elapsed_time(end)
            time_dict[f"block{i}"] = time_dict_mini


        # forward the entire sequence through the lm_head
        start.record()
        logits = self.lm_head(x)
        end.record()
        torch.cuda.synchronize()
        time_dict["lm_head"] = start.elapsed_time(end)

        input(time_dict)
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

        

