"""
A collection of attention layers with support for different positional encodings.
"""

import math
import torch
from models.components.positional_encoding import (
    LearnedPosEncoding,
    IdentityEncoding,
    SinCosPosEncoding,
    AbsolutePositionalEncoding,
    ALiBiPosEncoding,
    SANDWICHPosEncoding,
    xPOSPosEncoding,
    TransformerXLRelativePosEncoding,
    T5RelativePosEncoding,
    ShawRelativePosEncoding,
    LearnedRelativePosEncoding,
    build_positional_encodings
)

from models.components.utils.attention_utils import (
    apply_attention
)

class Attention(torch.nn.Module):
    """
    Flexible attention module with support for different attention mechanisms
    and positional encodings.
    """

    def __init__(
        self,
        hidden_dim,
        num_q_heads,
        num_kv_heads,
        bias,
        attention_type,
        pos_encoding_cfg,
        context_window,
        is_causal,
    ):
        super().__init__()
        assert hidden_dim % num_kv_heads == 0, "Hidden dim must be divisible by num_kv_heads"
        assert num_kv_heads % num_q_heads == 0, "num_kv_heads must be divisible by num_q_heads"

        group_size = num_kv_heads // num_q_heads

        # Key, query, value projections for all heads
        self.c_attn = torch.nn.Linear(
            hidden_dim, hidden_dim + 2 * hidden_dim // group_size, bias=bias
        )

        # Output projection
        self.c_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Attention dropout
        self.attn_dropout = torch.nn.Dropout()

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = group_size
        self.is_causal = is_causal

        # Select attention mechanism
        self.attention_type = attention_type
        self.attn_dropout_p = self.attn_dropout.p

        # Initialize the positional encoding
        self.pos_encoding = build_positional_encodings(
            pos_encoding_cfg=pos_encoding_cfg,
            context_window=context_window,
            hidden_dim=hidden_dim,
            num_heads=num_q_heads
        )

    def forward(self, x, attention_mask=None):
        """
        Forward pass of the attention module.

        Args:
            x (Tensor): Input tensor of shape (B, S, H)
            attention_mask (Tensor, optional): Boolean mask of shape (B, S, S)
                where True indicates positions to be masked out.

        Returns:
            Tensor: Output tensor of shape (B, S, H)
        """
        B, S, H = x.size()
        num_grouped_heads = self.num_kv_heads // self.group_size
        group_hidden_dim = H // self.group_size

        # Apply absolute positional encoding to input embeddings if applicable
        if isinstance(self.pos_encoding, (LearnedPosEncoding, SinCosPosEncoding, AbsolutePositionalEncoding)):
            x = self.pos_encoding(x)  # x is now positionally encoded

        # Compute query, key, values
        qkv = self.c_attn(x)  # (B, S, H + 2 * H / group_size)
        q, k, v = qkv.split([H, group_hidden_dim, group_hidden_dim], dim=-1)
        q = q.view(B, S, self.num_q_heads, H // self.num_q_heads)
        k = k.view(B, S, num_grouped_heads, H // self.num_kv_heads)
        v = v.view(B, S, num_grouped_heads, H // self.num_kv_heads)

        # If using RoPE or similar, apply after projection
        if isinstance(self.pos_encoding, xPOSPosEncoding):
            q, k = self.pos_encoding(q, k)

        # Transpose for multi-head attention
        q = q.transpose(1, 2)  # (B, num_q_heads, S, head_dim)
        k = k.transpose(1, 2)  # (B, num_kv_heads, S, head_dim)
        v = v.transpose(1, 2)  # (B, num_kv_heads, S, head_dim)

        # Reshape k and v to match q's number of heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Apply attention mechanism
        y = apply_attention(
            attention_mechanism_type=self.attention_type,
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            is_causal=self.is_causal,
            dropout_p=self.attn_dropout_p
        )  # (B, num_q_heads, S, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, S, H)  # Re-assemble all head outputs

        # Output projection
        y = self.attn_dropout(self.c_proj(y))

        return y


def build_attention(hidden_dim, context_window, attn_cfg):
    """
    Build an attention layer.

    Args:
        hidden_dim (int): Hidden dimension size.
        context_window (int): Context window size.
        attn_cfg (dict): Attention configuration dictionary.

    Returns:
        Attention: Configured attention layer.
    """
    is_causal = attn_cfg.get("is_causal", True)
    attention_type = attn_cfg.get("attention_mechanism", "standard")
    pos_encoding_cfg = attn_cfg.get("pos_enc_cfg", {"positional_encoding_type": "none"})
    num_kv_heads = attn_cfg["num_kv_heads"]
    num_q_heads = attn_cfg.get("num_q_heads", num_kv_heads)
    bias = attn_cfg.get("bias", False)

    attention_layer = Attention(
        hidden_dim=hidden_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        bias=bias,
        attention_type=attention_type,
        pos_encoding_cfg=pos_encoding_cfg,
        context_window=context_window,
        is_causal=is_causal,
    )
    return attention_layer
