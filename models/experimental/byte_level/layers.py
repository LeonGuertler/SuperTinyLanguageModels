"""
Shared components of the byte level models.
"""

import torch

from models.components.activations import build_activation
from models.components.attention import Attention
from models.components.normalization import build_normalization


class ProjectingFFN(torch.nn.Module):
    """
    A simple feedforward network
    """

    def __init__(
        self,
        hidden_dim,
        output_dim,
        ffn_dim,
        bias,
        ffn_activation,
    ):
        super().__init__()
        # build the ffn block
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.activation = build_activation(activation_name=ffn_activation)

        self.linear_2 = torch.nn.Linear(ffn_dim, output_dim, bias=bias)

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class ByteLevelTransformerBlock(torch.nn.Module):
    """
    A simple transformer block that combines
    FFN, Attn and normalization.
    """

    def __init__(self, input_dim, output_dim, ffn_dim, context_window, use_rope=False):
        super().__init__()

        # build the attn norm
        self.attn_norm = build_normalization(
            normalization_name="rms_norm", dim=input_dim, bias=False
        )

        # build the attention
        self.attn = Attention(
            hidden_dim=input_dim,
            num_heads=8,
            bias=False,
            use_rope=use_rope,
            context_window=context_window,
            is_causal=False,
            group_size=1,
        )

        # build the ffn norm
        self.ffn_norm = build_normalization(
            normalization_name="rms_norm", dim=input_dim, bias=False
        )

        # build the ffn block
        self.ffn = ProjectingFFN(
            hidden_dim=input_dim,
            ffn_dim=ffn_dim,
            output_dim=output_dim,
            bias=False,
            ffn_activation="gelu",
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
        x = self.ffn(self.ffn_norm(x))
        return x
