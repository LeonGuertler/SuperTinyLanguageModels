"""
A collection of FFN blocks
"""

import enum

import pydantic
import torch
import torch.nn.functional as F

from models.components.layers.activations import build_activation
from models.components.layers.normalization import build_normalization


class FFNTypes(str, enum.Enum):
    """
    Types of FFNs
    """

    GENERIC = "generic"
    SWIGLU = "swiglu"


class FFNConfig(pydantic.BaseModel):
    """
    Feedforward network configuration
    """

    ffn_dim: int
    bias: bool
    ffn_type: FFNTypes
    normalization: str


class GenericFFNConfig(FFNConfig):
    """
    Feedforward network configuration
    """

    ffn_type: FFNTypes.GENERIC
    ffn_activation: str


class SwiGLUFFNConfig(FFNConfig):
    """
    Feedforward network configuration
    """

    ffn_type: FFNTypes.SWIGLU


class GenericFFN(torch.nn.Module):
    """
    A simple feedforward network
    """

    def __init__(
        self,
        hidden_dim,
        ffn_config: GenericFFNConfig,
    ):
        super().__init__()
        # build the ffn block
        self.linear_1 = torch.nn.Linear(
            hidden_dim, ffn_config.ffn_dim, bias=ffn_config.bias
        )

        self.activation = build_activation(activation_name=ffn_config.ffn_activation)

        self.linear_2 = torch.nn.Linear(
            ffn_config.ffn_dim, hidden_dim, bias=ffn_config.bias
        )
        self.normalization = build_normalization(
            normalization_name=ffn_config.normalization,
            dim=hidden_dim,
            bias=ffn_config.bias,
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


class SwiGLUFFN(torch.nn.Module):
    """
    Implementation based on:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    originally from https://arxiv.org/abs/2002.05202

    N.B. does not support dropout
    """

    def __init__(
        self,
        hidden_dim,
        ffn_config: FFNConfig,
    ):
        super().__init__()
        # build the linear functions
        ffn_dim, bias = ffn_config.ffn_dim, ffn_config.bias
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.linear_2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=bias)

        self.linear_3 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.normalization = build_normalization(
            normalization_name=ffn_config.normalization,
            dim=hidden_dim,
            bias=ffn_config.bias,
        )

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.normalization(x)
        return self.linear_2(F.silu(self.linear_1(x)) * self.linear_3(x))


def build_ffn_config(ffn_cfg) -> FFNConfig:
    """
    Build the FFN config
    """
    match ffn_cfg["ffn_type"]:
        case FFNTypes.GENERIC:
            return GenericFFNConfig(**ffn_cfg)
        case FFNTypes.SWIGLU:
            return SwiGLUFFNConfig(**ffn_cfg)


def build_ffn(hidden_dim, ffn_cfg):
    """
    Build a feedforward network
    """
    ffn_config = build_ffn_config(ffn_cfg)
    match ffn_config.ffn_type:
        case FFNTypes.GENERIC:
            return GenericFFN(hidden_dim=hidden_dim, ffn_config=ffn_config)
        case FFNTypes.SWIGLU:
            return SwiGLUFFN(hidden_dim=hidden_dim, ffn_config=ffn_config)
