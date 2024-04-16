"""
A simple sequence to sequence model
"""
import math
import torch 
import torch.nn as nn


from models.layers import (
    LayerNorm,
    SelfAttention,
    CrossAttention,
    FFN,
    NextSequenceHead,
    StandardBlock
)

from models.embedding import (
    BaselineEmbedder
)


from models.weight_init import gpt2_weights_init
from models.utils import print_model_stats



class Seq2SeqModel(nn.Module):
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
                    [StandardBlock(
                        hidden_dim=cfg["hidden_dim"], 
                        ffn_dim=cfg["ffn_dim"], 
                        bias=cfg["bias"], 
                        num_heads=cfg["num_heads"], 
                        dropout=cfg["dropout"],
                    ) for _ in range(cfg["depth"])]
                )
            )
        )

        self.lm_head = NextSequenceHead(
            hidden_dim=cfg["hidden_dim"],
            vocab_size=cfg["vocab_size"]
        )

        # share the weights between the token embeddings and the final logit layer
        self.embedder.embedding.weight = (
            self.lm_head.linear.weight
        ) # https://paperswithcode.com/method/weight-tying

        # init all weights using GPT_2 weight init
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
