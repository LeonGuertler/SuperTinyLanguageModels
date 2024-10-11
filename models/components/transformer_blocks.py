"""
A collection of transformer blocks that combine
FFN, Attn and normalizatio
"""

import torch

from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from models.components.normalization import build_normalization


class GenericTransformerBlock(torch.nn.Module):
    """
    A simple transformer block that combines
    FFN, Attn and normalization.
    """

    def __init__(self, attention_type, hidden_dim, context_window, use_rope, ffn_cfg, attn_cfg):
        super().__init__()

        # build the attn norm
        self.attn_norm = build_normalization(
            normalization_name=attn_cfg["normalization"],
            dim=hidden_dim,
            bias=attn_cfg["bias"],
        )

        # build the attention
        self.attn = build_attention(
            attention_type=attention_type,
            hidden_dim=hidden_dim,
            context_window=context_window,
            use_rope=use_rope,
            attn_cfg=attn_cfg,
        )

        # build the ffn norm
        self.ffn_norm = build_normalization(
            normalization_name=ffn_cfg.get("normalization", "rms_norm"), # Default: rms_norm
            dim=hidden_dim,
            bias=ffn_cfg["bias"],
        )

        # build the ffn block
        self.ffn = build_ffn(
            hidden_dim=hidden_dim,
            ffn_cfg=ffn_cfg,
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
            attention_mask: the attention mask
        Returns:
            x: the output tensor (b, s, h)
        """
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
