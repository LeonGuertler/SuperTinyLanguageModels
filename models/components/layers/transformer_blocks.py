"""
A collection of transformer blocks that combine
FFN, Attn and normalizatio
"""

import torch

from models.components.layers.attention import AttentionConfig, build_attention
from models.components.layers.feedforward import FFNConfig, build_ffn


class GenericTransformerBlock(torch.nn.Module):
    """
    A simple transformer block that combines
    FFN, Attn and normalization.
    """

    def __init__(
        self, hidden_dim, context_window, ffn_cfg: FFNConfig, attn_cfg: AttentionConfig
    ):
        super().__init__()
        attn_cfg = AttentionConfig(**attn_cfg)

        # build the attention
        self.attn = build_attention(
            hidden_dim=hidden_dim,
            context_window=context_window,
            attn_cfg=attn_cfg,
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
        x = x + self.attn(x, attention_mask)
        x = x + self.ffn(x)
        return x
