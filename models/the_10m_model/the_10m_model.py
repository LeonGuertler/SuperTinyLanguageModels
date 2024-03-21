import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# import the layers
from models.layers import (
    LayerNorm,
    CausalSelfAttention,
    FFN
)

from models.tokenizer import character_bpe_tokenizer



class the10mmodel(nn.Module):
    def __init__(self, config):
        self.config = config

        # load the tokenizer
        self.tokenizer = character_bpe_tokenizer(
            config=config
        )

        # prepare the dataset if necessary
        self.tokenizer.prepare_dataset()