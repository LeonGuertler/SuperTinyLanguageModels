"""
Init to simplify imports.
"""
import torch 
import torch.nn as nn

from models.components.layers.activations import (
    build_activation
)

from models.components.layers.normalization import (
    LayerNorm
)

from models.components.layers.attention import (
    CausalSelfAttention,
)

class BaseTransformerBlock(nn.Module):
    """
    A simple abstraction to combine the 
    LayerNorms, SelfAttention and FeedForward layers
    """
    def __init__(self, hidden_dim, ffn_dim, ffn_activation, bias, num_heads, dropout):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = FFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            dropout=dropout,
            ffn_activation=ffn_activation
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward 
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class FFN(nn.Module):
    """
    A simple Feed Forward Network block.
    """
    def __init__(self, hidden_dim, ffn_dim, bias=False, dropout=0.0, ffn_activation:str="gelu"):
        super().__init__()
        self.c_fc = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=bias,
        )

        self.gelu = build_activation(
            activation_name=ffn_activation
        )
        self.c_proj = nn.Linear(
            ffn_dim,
            hidden_dim,
            bias=bias,
        )
        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        """
        Forward pass
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x