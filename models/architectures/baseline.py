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
from models.layers import LayerNorm, CausalSelfAttention, FFN, Block, NextTokenHead

from models.embedding import BaselineEmbedder

from models.weight_init import gpt2_weights_init
from models.utils import print_model_stats


class BaseGPT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg["vocab_size"] is not None
        assert cfg["context_window"] is not None

        self.cfg = cfg

        # construct the actual model
        # these can be overriden in the child class
        self.embedder = self.build_embedder()
        self.transformer = self.build_transformer()
        self.lm_head = self.build_lm_head()
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

    def build_embedder(self):
        return BaselineEmbedder(
            hidden_dim=self.cfg["hidden_dim"],
            context_window=self.cfg["context_window"],
            vocab_size=self.cfg["vocab_size"],
        )

    def build_transformer(self):
        return nn.ModuleDict(
            dict(
                drop=nn.Dropout(self.cfg["dropout"]),
                h=nn.ModuleList(
                    [Block(
                        hidden_dim=self.cfg["hidden_dim"], 
                        ffn_dim=self.cfg["ffn_dim"], 
                        bias=self.cfg["bias"], 
                        num_heads=self.cfg["num_heads"], 
                        dropout=self.cfg["dropout"],
                    ) for _ in range(self.cfg["depth"])]
                )
            )
        )

    def build_lm_head(self):
        return NextTokenHead(
            hidden_dim=self.cfg["hidden_dim"],
            vocab_size=self.cfg["vocab_size"],
        )

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

        

