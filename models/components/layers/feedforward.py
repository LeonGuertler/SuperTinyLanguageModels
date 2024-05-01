"""
A collection of different FFN blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.layers.activations import build_activation
from models.components.layers.moe import MoE


class FFN(nn.Module):
    """
    A simple Feed Forward Network block.
    """

    def __init__(self, hidden_dim, ffn_dim, bias=False, ffn_activation: str = "gelu"):
        super().__init__()
        self.c_fc = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=bias,
        )

        self.gelu = build_activation(activation_name=ffn_activation)
        self.c_proj = nn.Linear(
            ffn_dim,
            hidden_dim,
            bias=bias,
        )
        self.dropout = nn.Dropout()

    def forward(self, x):
        """
        Forward pass
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SWIGluFFN(nn.Module):
    """
    Implementation based on:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    originally from https://arxiv.org/abs/2002.05202

    N.B. does not support dropout #TODO it should?
    """

    def __init__(self, hidden_dim, ffn_dim, **_):
        super().__init__()

        self.lin_1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.lin_2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.lin_3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x):
        """
        Forward pass
        """
        return self.lin_2(F.silu(self.lin_1(x)) * self.lin_3(x))


class JetMoEFFN(nn.Module):
    """
    Implementation based on: https://github.com/myshell-ai/JetMoE/blob/main/jetmoe/modeling_jetmoe.py
    """

    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k, bias):
        super().__init__()
        self.mlp = MoE(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias,
        )

    def forward(self, x):
        """Foward pass"""
        x_mlp, mlp_aux_loss = self.mlp(x)
        return x_mlp, mlp_aux_loss


class BSpline(nn.Module):
    def __init__(self, hidden_dim, grid_intervals_init, order=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.order = order
        self.grid_intervals = nn.Parameter(
            torch.tensor(grid_intervals_init, dtype=torch.float32)
        )

        # Make the knot vector learnable
        self.knot_vector = nn.Parameter(
            torch.linspace(-1, 1, grid_intervals_init + order + 1)
        )

    def basis_function(self, u, i, p, U):
        """
        Computes the B-spline basis function of degree p for the knot span i.
        """
        if p == 0:
            return torch.where((U[i] <= u) & (u < U[i + 1]), 1.0, 0.0)
        else:
            term1 = (
                0.0
                if U[i] == U[i + p]
                else ((u - U[i]) / (U[i + p] - U[i]))
                * self.basis_function(u, i, p - 1, U)
            )
            term2 = (
                0.0
                if U[i + p + 1] == U[i + 1]
                else ((U[i + p + 1] - u) / (U[i + p + 1] - U[i + 1]))
                * self.basis_function(u, i + 1, p - 1, U)
            )
            return term1 + term2

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        grid_intervals = int(self.grid_intervals.item())  # Convert to int for indexing

        x = x.view(batch_size * seq_len, self.hidden_dim)
        basis = torch.stack(
            [
                self.basis_function(x, i, self.order, self.knot_vector)
                for i in range(grid_intervals)
            ],
            dim=2,
        )
        x = torch.bmm(
            x.unsqueeze(1),
            basis.view(batch_size * seq_len, grid_intervals, self.order + 1),
        ).squeeze(1)
        return x.view(batch_size, seq_len, self.hidden_dim)


class KAN(nn.Module):
    """Drop-in replacement of Linear layer with KAN layer
    From paper: https://arxiv.org/pdf/2404.19756
    - KAN: Kolmogorov–Arnold Networks

    These have weight-wise non-linearity defined by:
    $$\phi(x)= \text{swish}(x)+\text{spline}(x)$$"""

    def __init__(self, hidden_dim, ffn_dim, bias=False, order=3, grid_intervals=4):
        super().__init__()
        self.cs = nn.ParameterList(
            [nn.Parameter(torch.randn(ffn_dim, hidden_dim)) for _ in range(order)]
        )

        # We have order no. of splines
        self.b_splines = nn.ModuleList(
            [BSpline(hidden_dim, grid_intervals) for _ in range(order)]
        )
        self.order = order

    def forward(self, x):
        # x: (b, s, h)
        # first compute the swish part
        swish = F.silu(x)

        # then compute the spline part
        spline = 0
        for i in range(self.order):
            spline += self.b_splines[i](self.cs[i] @ x.transpose(1, 2)).transpose(1, 2)

        return swish + spline


class KANFFN(nn.Module):
    """KAN Feed Forward Network with linear layers

    replaced with KAN layers"""

    def __init__(self, hidden_dim, ffn_dim, bias=False, order=3, grid_intervals=4, **_):
        super().__init__()
        self.k_in = KAN(hidden_dim, ffn_dim, order=order, grid_intervals=grid_intervals)
        self.k_out = KAN(
            ffn_dim, hidden_dim, order=order, grid_intervals=grid_intervals
        )

    def forward(self, x):
        x = self.k_in(x)
        x = self.k_out(x)
        return x


def build_ffn(ffn_type: str, **kwargs):
    """
    Build the FFN block based on the name
    Options:
        - ffn
        - swiglu
        - jetmoe
    """
    if ffn_type == "ffn":
        return FFN(**kwargs)
    elif ffn_type == "swiglu":
        return SWIGluFFN(**kwargs)
    elif ffn_type == "jetmoe":
        return JetMoEFFN(**kwargs)
    elif ffn_type == "kan":
        return KANFFN(**kwargs)
    else:
        raise ValueError(f"Unknown FFN block: {ffn_type}")
