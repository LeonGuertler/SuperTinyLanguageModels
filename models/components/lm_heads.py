"""
A collection of Language Model heads.
"""

import torch.nn as nn

from models.components.layers import LayerNorm


class LMHead(nn.Module):
    """Interface for the Language Model head."""

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError

    def inference(self, x):
        """Inference."""
        x = x[:, -1, :]
        return self(x)


class NextTokenHead(LMHead):
    """Next token prediction head."""

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.ln = LayerNorm(hidden_dim, bias=True)
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        """Forward pass."""
        x = self.ln(x)
        logits = self.linear(x)
        return logits


def build_lm_head(head_type, hidden_dim, vocab_size):
    """Build a language model head."""
    if head_type == "next_token":
        return NextTokenHead(hidden_dim, vocab_size)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
