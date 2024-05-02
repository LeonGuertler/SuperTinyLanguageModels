"""Predefined Blocks for use out of the box."""

import torch.nn as nn

from models.components.layers.attention import SelfAttention
from models.components.layers.feedforward import FFN, SWIGluFFN
from models.components.layers.moe import MoE
from models.components.layers.normalization import LayerNorm, build_normalization


class BaseTransformerBlock(nn.Module):
    """
    A simple abstraction to combine the
    LayerNorms, SelfAttention and FeedForward layers
    """

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        ffn_activation,
        bias,
        num_heads,
        normalization="layernorm",
    ):
        super().__init__()
        self.norm_1 = build_normalization(
            normalization,
            hidden_dim,
            bias=bias,
        )
        self.attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            is_causal=True,
        )
        self.norm_2 = build_normalization(
            normalization,
            hidden_dim,
            bias=bias,
        )
        self.mlp = FFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            ffn_activation=ffn_activation,
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.norm_1(x), attention_mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class BidirectionalTransformerBlock(nn.Module):
    """
    A simple abstraction to combine the
    LayerNorms, SelfAttention and FeedForward layers
    """

    def __init__(self, hidden_dim, ffn_dim, ffn_activation, bias, num_heads):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            is_causal=False,
            bias=bias,
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = FFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            ffn_activation=ffn_activation,
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ModernTransformerBlock(nn.Module):
    """
    A simple abstraction to combine the
    RMSNorm, SelfAttention (RoPE) and ModernFFN Layers
    """

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        num_heads,
        context_window,
        normalization="rmsnorm",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_context_window=context_window,
            use_rope=True,
        )

        self.ffn = SWIGluFFN(hidden_dim=hidden_dim, ffn_dim=ffn_dim)

        self.attn_norm = build_normalization(
            normalization,
            hidden_dim,
        )
        self.ffn_norm = build_normalization(
            normalization,
            hidden_dim,
        )

    def forward(self, x):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.attn_norm(x))  # freqs_cis?
        x = x + self.ffn(self.ffn_norm(x))
        return x


class JetFFNMoEBlock(nn.Module):
    """
    A MoE block based on JetMoE, but
    with standard causal attention.
    """

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        num_heads,
        context_window,
        num_experts,
        top_k,
    ):
        super().__init__()
        self.attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_context_window=context_window,
            use_rope=True,
        )

        self.ffn = MoE(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            bias=False,
        )

        self.attn_norm = build_normalization(
            "rmsnorm",
            hidden_dim,
        )
        self.ffn_norm = build_normalization(
            "rmsnorm",
            hidden_dim,
        )

    def forward(self, x):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.attn_norm(x))  # freqs_cis?
        h, moe_aux_loss = self.ffn(self.ffn_norm(x))
        x = x + h
        return x, moe_aux_loss