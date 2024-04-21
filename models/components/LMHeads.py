"""
A collection of Language Model heads.
"""

import torch
import torch.nn as nn


from models.components.layers import (
    LayerNorm
)

class NextTokenHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.ln = LayerNorm(hidden_dim, bias=True)
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.ln(x)
        logits = self.linear(x)
        return logits