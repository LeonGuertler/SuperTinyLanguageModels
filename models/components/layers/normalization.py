"""
A collection of normalization layers.
"""

import torch
from torch.nn import functional as F


class LayerNorm(torch.nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    # taken from nanoGPT
    def __init__(self, dim, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        """Apply Layer Norm"""
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(torch.nn.Module):
    """
    RMSNorm (https://arxiv.org/abs/1910.07467), implementation from
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMSNorm"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


NORMALIZATION_DICT = {
    "rms_norm": lambda dim, bias: RMSNorm(dim=dim),
    "layer_norm": lambda dim, bias: LayerNorm(dim=dim, bias=bias),
    "none": lambda dim, bias: torch.nn.Identity(),
}


def build_normalization(normalization_name, dim, bias=None):
    """
    Build the normalization layer
    Available options: rmsnorm, layernorm
        - Bias is ignored for RMSNorm
    """
    return NORMALIZATION_DICT[normalization_name](dim=dim, bias=bias)
