"""Implements the modern baseline described in https://arxiv.org/pdf/2312.00752.pdf (Mamba)"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

# import the layers
from models.layers import (
    SwiGLUFNN,
    RMSNorm,
    SelfAttention, 
    ModernBlock,
    NextTokenHead
)

from models.embedding import BaselineEmbedder

from models.weight_init import gpt2_weights_init
from models.utils import print_model_stats, precompute_freqs_cis

    

class ModernGPT(nn.Module):

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
            tokenizer_name=cfg["tokenizer"],
            positional_encoder_type=cfg["positional_encoder"]
        )
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(cfg["dropout"]),
                h=nn.ModuleList(
                    [ModernBlock(
                        hidden_dim=cfg["hidden_dim"], 
                        ffn_dim=cfg["ffn_dim"], 
                        bias=cfg["bias"], 
                        num_heads=cfg["num_heads"], 
                        dropout=cfg["dropout"],
                        apply_rope=cfg["positional_encoder"]=="rope"
                    ) for _ in range(cfg["depth"])]  
                )
            )
        )
        self.rope_freqs = precompute_freqs_cis(dim=cfg["hidden_dim"], end=cfg["context_window"])

        self.lm_head = NextTokenHead(
            hidden_dim=cfg["hidden_dim"],
            vocab_size=cfg["vocab_size"],
        )

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
        rope_freqs = self.rope_freqs.to(x.device)
        for block in self.transformer.h:
            x = block(x, rope_freqs=rope_freqs)

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
        for block in self.transformer.h:
            x = block(x)

        # forward only the last token through the lm_head
        logits = self.lm_head(x[:, -1, :])
        return logits

        

