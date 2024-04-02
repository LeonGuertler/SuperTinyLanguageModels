"""
A collection of positional encoding modules.
"""

import torch
import torch.nn as nn


class LearnedPosEncoding(nn.Module):
    """
    Basic learned positional encoding
    """
    def __init__(self, hidden_dim, context_window):
        super().__init__()
        self.pe = nn.Embedding(context_window, hidden_dim)

    def forward(self, x):
        """
        Forward pass
        """
        return self.pe(torch.arange(x.size(1), device=x.device))