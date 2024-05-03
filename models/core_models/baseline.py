"""
GPT-2 like transformer.
"""

import torch.nn as nn

from models.components.layers import BaseTransformerBlock
from models.components.positional_encoding import LearnedPosEncoding


class StandardTransformer(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the standard transformer model
        similar to gpt-2
        """
        super().__init__()

        self.core_model_cfg = cfg["core_model"]

        # build positional encoding
        self.pos_encoder = LearnedPosEncoding(
            hidden_dim=cfg["core_model"]["hidden_dim"],
            context_window=cfg["model_shell"]["context_window"],
        )

        # build the transformer
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(),
                h=nn.ModuleList(
                    [
                        BaseTransformerBlock(
                            hidden_dim=self.core_model_cfg["hidden_dim"],
                            ffn_dim=self.core_model_cfg["ffn_dim"],
                            ffn_activation=self.core_model_cfg["ffn_activation"],
                            bias=self.core_model_cfg["bias"],
                            num_heads=self.core_model_cfg["num_heads"],
                        )
                        for _ in range(self.core_model_cfg["depth"])
                    ]
                ),
            )
        )

    def forward(self, x):
        """
        Pass an input through the model
        """
        # positional encoding
        x = x + self.pos_encoder(x)

        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)

        return x
