import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# import the layers
<<<<<<< HEAD
from models import layers
from models import baseline
=======
from models.layers import LayerNorm, CausalSelfAttention, FFN

>>>>>>> main
from models.tokenizer import tokenizer


class Block(nn.Module):

    def __init__(self, config, shared_mlp_block):
        super().__init__()
<<<<<<< HEAD
        self.ln_1 = layers.LayerNorm(
            config["arch"]["hidden_dim"], bias=config["arch"]["bias"]
        )
        self.attn = layers.CausalSelfAttention(config)
        self.ln_2 = layers.LayerNorm(
            config["arch"]["hidden_dim"], bias=config["arch"]["bias"]
        )
        self.mlp = layers.FFN(config)
=======
        self.ln_1 = LayerNorm(config["arch"]["hidden_dim"], bias=config["arch"]["bias"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config["arch"]["hidden_dim"], bias=config["arch"]["bias"])
        self.mlp = FFN(config)
>>>>>>> main

        # share the mlp block
        self.mlp.c_fc.weight = shared_mlp_block.c_fc.weight
        self.mlp.c_fc.bias = shared_mlp_block.c_fc.bias
        self.mlp.c_proj.weight = shared_mlp_block.c_proj.weight
        self.mlp.c_proj.bias = shared_mlp_block.c_proj.bias

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FFNShareGPT(baseline.BaseGPT):

    def __init__(self, config):
<<<<<<< HEAD
        super().__init__(config)
=======
        super().__init__()
>>>>>>> main
        assert config["arch"]["vocab_size"] is not None
        assert config["arch"]["context_window"] is not None
        self.config = config
        self.tokenizer = tokenizer(config=config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer.device = self.device

        # prepare the dataset if necessary
        self.tokenizer.prepare_dataset()

        self.shared_mlp_block = layers.FFN(config)
        # construct the actual model
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config["arch"]["vocab_size"], config["arch"]["hidden_dim"]
                ),
                wpe=nn.Embedding(
                    config["arch"]["context_window"], config["arch"]["hidden_dim"]
                ),
                drop=nn.Dropout(config["arch"]["dropout"]),
                h=nn.ModuleList(
                    [
                        Block(config, shared_mlp_block=self.shared_mlp_block)
                        for _ in range(config["arch"]["depth"])
                    ]
                ),
<<<<<<< HEAD
                ln_f=layers.LayerNorm(
=======
                ln_f=LayerNorm(
>>>>>>> main
                    config["arch"]["hidden_dim"], bias=config["arch"]["bias"]
                ),
            )
        )
        self.lm_head = nn.Linear(
            config["arch"]["hidden_dim"], config["arch"]["vocab_size"], bias=False
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config["arch"]["depth"])
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
