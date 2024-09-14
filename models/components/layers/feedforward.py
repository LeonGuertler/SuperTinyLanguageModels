"""
A collection of FFN blocks
"""

import torch
import torch.nn.functional as F

from models.components.layers.activations import build_activation


class GenericFFN(torch.nn.Module):
    """
    A simple feedforward network
    """

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        bias,
        ffn_activation,
        ffn_dropout
    ):
        super().__init__()
        # build the ffn block
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.activation = build_activation(activation_name=ffn_activation)

        self.linear_2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=bias)

        self.dropout = torch.nn.Dropout(
            p=ffn_dropout
        )

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class SiluFFN(torch.nn.Module):
    """
    Implementation based on:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    originally from https://arxiv.org/abs/2002.05202

    N.B. does not support dropout
    """

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        bias,
        ffn_dropout
    ):
        super().__init__()
        # build the linear functions
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.linear_2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=bias)

        self.linear_3 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.dropout = torch.nn.Dropout(
            p=ffn_dropout
        )

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.dropout(x)
        return self.linear_2(F.silu(self.linear_1(x)) * self.linear_3(x))


FFN_DICT = {
    "generic": lambda hidden_dim, ffn_cfg: GenericFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        bias=ffn_cfg.get("bias", False), # Default to False
        ffn_activation=ffn_cfg.get("activation", "gelu"), # Default to 'gelu
        ffn_dropout=ffn_cfg.get("ffn_dropout", 0.0) # Default to 0.0
    ),
    "silu_ffn": lambda hidden_dim, ffn_cfg: SiluFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        bias=ffn_cfg.get("bias", False), # Default to False
        ffn_dropout=ffn_cfg.get("ffn_dropout", 0.0) # Default to 0.0
    ),
}


def build_ffn(hidden_dim, ffn_cfg):
    """
    Build a feedforward network
    """
    assert ffn_cfg["ffn_type"] in FFN_DICT, \
        f"FFN type {ffn_cfg['ffn_type']} not found. Available types: {FFN_DICT.keys()}"
    
    return FFN_DICT[ffn_cfg["ffn_type"]](hidden_dim=hidden_dim, ffn_cfg=ffn_cfg)
