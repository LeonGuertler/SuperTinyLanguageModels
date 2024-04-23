"""
Llama-3 like transformer.
"""

import torch.nn as nn

from models.components.positional_encoding import LearnedPosEncoding

from models.components.layers import ModernTransformerBlock


class ModernFFNSharingTransformer(nn.Module):
    def __init__(self, cfg):
        """
        Initialize a Llama-3 style transformer model
        including:
            - rope
            - SwiGLU
            - RMSNorm
        """
        super().__init__()

        self.core_model_cfg = cfg["core_model"]
        self.context_window = cfg["model_shell"]["context_window"]

        # build positional encoding
        self.pos_encoder = LearnedPosEncoding(
            hidden_dim=cfg["core_model"]["hidden_dim"],
            context_window=self.context_window,
        )

        # build the transformer
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(self.core_model_cfg["dropout"]),
                h=nn.ModuleList(
                    [
                        ModernTransformerBlock(
                            hidden_dim=self.core_model_cfg["hidden_dim"],
                            ffn_dim=self.core_model_cfg["ffn_dim"],
                            num_heads=self.core_model_cfg["num_heads"],
                            dropout=self.core_model_cfg["dropout"],
                            context_window=self.context_window,
                        )
                        for _ in range(self.core_model_cfg["depth"])
                    ]
                ),
            )
        )

        # share the ffn weights between the layers
        for block in self.transformer.h:
            # done individually to tie rather than share
            block.mlp.lin_1.weight = self.transformer.h[0].mlp.lin_1.weight
            block.mlp.lin_2.weight = self.transformer.h[0].mlp.lin_2.weight
            block.mlp.lin_3.weight = self.transformer.h[0].mlp.lin_3.weight

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
