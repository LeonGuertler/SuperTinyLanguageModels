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




class Seq2SeqModelShell(nn.Module):
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
        self.tokenizer = build_tokenizer(
            tokenizer_type=self.cfg["model_shell"]["tokenizer"],
            vocab_size=self.cfg["model_shell"]["vocab_size"],
            dataset_name=self.cfg["model_shell"]["tokenizer_dataset_name"],
        )

        # build the embedder
        self.vae = core_model


        # report number of parameters
        print_model_stats(self)

        # gpt-2 weight init
        self.apply(self._init_weights)


    def forward(self, token_ids):
        """
        The default forward pass is used for training and accepts the 
        token_ids as input. When the model is in eval mode, only the 
        last token is passed into the NextTokenHead.
        """

        return self.vae(token_ids)
