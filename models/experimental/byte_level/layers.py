"""
Shared components of the byte level models.
"""

import pydantic
import torch

from models.components.layers.activations import build_activation
from models.components.layers.attention import AttentionConfig, build_attention
from models.components.layers.normalization import build_normalization


class ByteTransformerBlockConfig(pydantic.BaseModel):
    """
    Feedforward network configuration
    """

    bias: bool
    ffn_activation: str
    ffn_normalization: str = "rmsprop"


class ProjectingFFN(torch.nn.Module):
    """
    A simple feedforward network
    """

    def __init__(
        self,
        input_dim,
        ffn_dim,
        output_dim,
        block_cfg: ByteTransformerBlockConfig,
    ):
        super().__init__()
        # build the ffn block
        self.linear_1 = torch.nn.Linear(input_dim, ffn_dim, bias=block_cfg.bias)

        self.activation = build_activation(activation_name=block_cfg.ffn_activation)

        self.linear_2 = torch.nn.Linear(ffn_dim, output_dim, bias=block_cfg.bias)
        self.normalization = build_normalization(
            normalization_name="rmsprop", dim=input_dim, bias=block_cfg.bias
        )

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.normalization(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class ByteLevelTransformerBlock(torch.nn.Module):
    """
    A simple transformer block that combines
    FFN, Attn and normalization.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        ffn_dim,
        context_window,
        byte_transformer_block_cfg: ByteTransformerBlockConfig,
        attn_config: AttentionConfig,
    ):
        super().__init__()

        # build the attention
        self.attn = build_attention(
            hidden_dim=input_dim, context_window=context_window, attn_cfg=attn_config
        )

        # build the ffn block
        self.ffn = ProjectingFFN(
            input_dim=input_dim,
            ffn_dim=ffn_dim,
            output_dim=output_dim,
            block_cfg=byte_transformer_block_cfg,
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
        x = self.ffn(x)
        return x
