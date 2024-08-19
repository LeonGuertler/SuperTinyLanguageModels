"""
A collection of normalization layers.
"""

import enum

import pydantic
import torch
from torch.nn import functional as F

EPSILON = 1e-6


class NormalizationTypes(str, enum.Enum):
    """
    Types of normalization
    """

    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"


class NormConfig(pydantic.BaseModel):
    """
    Normalization configuration
    """

    normalization: NormalizationTypes
    dim: pydantic.PositiveInt


class LayerNormConfig(NormConfig):
    """
    Layer normalization configuration
    """

    normalization: NormalizationTypes.LAYERNORM
    bias: bool = True


class LayerNorm(torch.nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    # taken from nanoGPT
    def __init__(self, norm_config: LayerNormConfig):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(norm_config.dim))
        self.bias = (
            torch.nn.Parameter(torch.zeros(norm_config.dim))
            if norm_config.bias
            else None
        )

    def forward(self, x):
        """Apply Layer Norm"""
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNormConfig(NormConfig):
    """
    RMSNorm configuration

    eps is the epsilon value to prevent division by zero
    """

    normalization: NormalizationTypes.RMSNORM
    eps: pydantic.PositiveFloat = EPSILON


class RMSNorm(torch.nn.Module):
    """
    RMSNorm (https://arxiv.org/abs/1910.07467), implementation from
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """

    def __init__(self, norm_config: RMSNormConfig):
        super().__init__()
        self.eps = norm_config.eps
        self.weight = torch.nn.Parameter(torch.ones(norm_config.dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMSNorm"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def build_normalization(
    normalization_name: NormalizationTypes, dim: int, bias: bool = True
):
    """
    Build the normalization layer
    Available options: rmsnorm, layernorm
        - Bias is ignored for RMSNorm
    """
    match normalization_name:
        case NormalizationTypes.LAYERNORM:
            return LayerNorm(
                LayerNormConfig(normalization=normalization_name, dim=dim, bias=bias)
            )
        case NormalizationTypes.RMSNORM:
            return RMSNorm(RMSNormConfig(normalization=normalization_name, dim=dim))
        case _:
            raise ValueError(f"Unknown normalization type: {normalization_name}")
