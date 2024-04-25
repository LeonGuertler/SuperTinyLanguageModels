"""
Llama-3 like transformer.
"""

import torch.nn as nn

from models.components.layers import ModernTransformerBlock


class ModernTransformer(nn.Module):
    """Transformer representing the modern standard as used in e.g.
    Llama-3.
    """

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

        # build the transformer
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(),
                h=nn.ModuleList(
                    [
                        ModernTransformerBlock(
                            hidden_dim=self.core_model_cfg["hidden_dim"],
                            ffn_dim=self.core_model_cfg["ffn_dim"],
                            num_heads=self.core_model_cfg["num_heads"],
                            context_window=self.context_window,
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
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)

        return x
