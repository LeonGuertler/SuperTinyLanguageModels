"""
A collection of normalization layers.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    # taken from nanoGPT
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        """Apply Layer Norm"""
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """
    RMSNorm (https://arxiv.org/abs/1910.07467), implementation from
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMSNorm"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def build_normalization(normalization_name, hidden_dim, bias=None):
    """
    Build the normalization layer
    Available options: rmsnorm, layernorm
        - Bias is ignored for RMSNorm
    """
    if normalization_name == "rmsnorm":
        return RMSNorm(hidden_dim)
    elif normalization_name == "layernorm":
        return LayerNorm(hidden_dim, bias=bias)
    else:
        raise ValueError(f"Normalization {normalization_name} not supported.")
