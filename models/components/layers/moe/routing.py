"""
A collection of different moe routing functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGating(nn.Module):
    """
    Paper: https://arxiv.org/abs/1701.06538
    Implementation based on: https://github.com/myshell-ai/JetMoE/blob/main/jetmoe/utils/gate.py
    """

    def __init__(self, input_size, num_experts, top_k):
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts, "top_k should be less than or equal to num_experts"
        self.top_k = top_k
        self.loss = 0  # I hate this...

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def compute_aux_loss(self, probs, logits, gates):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.
        """
        count = logits.size(0)
        probs = probs.sum(0)
        freq = (gates > 0).float().sum(0)
        lsesq = (torch.log(torch.exp(logits).sum(dim=-1)) ** 2).sum()

        switchloss = (
            self.num_experts
            * (F.normalize(probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        )

        z_loss = lsesq / count
        loss = switchloss + 0.1 * z_loss

        return loss

    def forward(self, x):
        """Forward pass"""

        logits = self.layer(x).float()
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(x)

        if self.training:
            probs = torch.softmax(logits, dim=1)
            zeros = torch.zeros_like(probs)
            zeros = zeros.to(top_k_gates.dtype)
            gates = torch.scatter(1, top_k_indices, top_k_gates)
            self.loss = self.compute_aux_loss(probs, logits, gates)
        else:
            self.loss = 0

        return top_k_indices, top_k_gates
