"""
A collection of transformer blocks that combine
FFN, Attn and normalizatio
"""

import torch

from models.components.attention import build_attention
from models.components.feedforward import build_ffn

from typing import Optional


class GenericTransformerBlock(torch.nn.Module):
    """
    A simple transformer block that combines
    FFN, Attn and normalization.
    """

    def __init__(self, hidden_dim, context_window, ffn_cfg, attn_cfg, depth: Optional[int]=None):
        super().__init__()

        # build the attn norm
        # self.attn_norm = build_normalization(
        #     normalization_name=attn_cfg.get("normalization", "none"),
        #     dim=hidden_dim,
        #     bias=attn_cfg["bias"],
        # )

        # build the attention
        self.attn = build_attention(
            attn_name=attn_cfg["name"],
            attn_params=attn_cfg["params"],
            hidden_dim=hidden_dim,
            context_window=context_window,
            depth=depth,
        )

        # build the ffn norm
        # self.ffn_norm = build_normalization(
        #     normalization_name=ffn_cfg.get("normalization", "none"), # Default: none
        #     dim=hidden_dim,
        #     bias=ffn_cfg["bias"],
        # )

        # build the ffn block
        self.ffn = build_ffn(
            ffn_name=ffn_cfg["name"],
            ffn_params=ffn_cfg["params"],
            hidden_dim=hidden_dim,
        )

    def forward(self, x, attn_mask=None):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
            attn_mask: the attention mask
        Returns:
            x: the output tensor (b, s, h)
        """
        x = x + self.attn(x, attn_mask)
        x = x + self.ffn(x)

        #x = x + self.attn(self.attn_norm(x), attn_mask)
        #x = x + self.ffn(self.ffn_norm(x))
        return x
