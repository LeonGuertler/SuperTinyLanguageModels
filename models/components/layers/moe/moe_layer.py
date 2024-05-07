"""
A collection of different MOE implementations.
"""

import torch
import torch.nn as nn

from models.components.layers.activations import build_activation
from models.components.layers.moe.routing import TopKGating
from models.components.layers.moe.utils import ParallelExperts, compute_gating


class MoE(nn.Module):
    """
    Sparsley gated moe layer with 1-layer ffn as experts.
    Implementation based on: https://github.com/myshell-ai/JetMoE/blob/main/jetmoe/utils/moe.py
    """

    def __init__(
        self, hidden_dim, ffn_dim, num_experts, top_k, bias, activation="silu", glu=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias = bias
        self.activation = activation
        self.glu = glu

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(hidden_dim))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

        self.input_linear = ParallelExperts(
            num_experts, hidden_dim, ffn_dim * 2 if glu else ffn_dim
        )
        self.output_linear = ParallelExperts(num_experts, ffn_dim, hidden_dim)

        self.top_k = min(top_k, num_experts)
        self.activation = build_activation(activation_name=activation)
        self.input_size = hidden_dim

        self.router = TopKGating(
            input_size=self.input_size, num_experts=num_experts, top_k=top_k
        )

    def get_aux_loss_and_clear(self):
        """
        Get the auxiliary loss and clear the loss attribute.
        """
        return self.gate.get_aux_loss_and_clear()

    def _compute_gate(self, x):
        top_k_indices, top_k_gates = self.router(x)

        batch_gates, batch_index, expert_size, _ = compute_gating(
            self.top_k, self.num_experts, top_k_gates, top_k_indices
        )

        expert_size = expert_size.tolist()

        return self.router.loss, batch_gates, batch_index, expert_size

    def forward(self, x):
        """Forward pass the moe layer"""
        B, S, H = x.size()
        x = x.reshape(-1, H)
        loss, batch_gates, batch_index, expert_size = self._compute_gate(x)

        expert_inputs = x[batch_index]
        h = self.input_linear(expert_inputs, expert_size)
        if self.glu:
            h, g = h.chunk(2, dim=-1)
            h = self.activation(h) * g
        else:
            h = self.activation(h)

        expert_outputs = self.output_linear(h, expert_size)

        expert_outputs = expert_outputs * batch_gates[:, None]

        zeros = torch.zeros(
            (B * S, self.hidden_dim),
            dtype=expert_outputs.dtype,
            device=expert_outputs.device,
        )
        y = zeros.index_add(0, batch_index, expert_outputs)
        y = y.view(B, S, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y, loss
