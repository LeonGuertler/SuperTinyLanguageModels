"""
Init to simplify imports.
"""
import torch 
import torch.nn as nn


from models.components.layers.normalization import (
    LayerNorm,
    RMSNorm
)

from models.components.layers.attention import (
    CausalSelfAttention,
    RoPESelfAttention
)

from models.components.layers.feedforward import (
    FFN,
    LLama3FFN
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
    

class Llama3TransformerBlock(nn.Module):
    """
    A simple abstraction to combine the 
    RMSNorm, SelfAttention (RoPE) and LLama3FFNblock
    """
    def __init__(self, hidden_dim, ffn_dim, num_heads, dropout, context_window):
        super().__init__()
        self.num_heads = num_heads 
        self.hidden_dim = hidden_dim 

        self.attn = RoPESelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            context_window=context_window,
            dropout=dropout,
        )

        self.ffn = LLama3FFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim
        )

        self.attn_norm = RMSNorm(
            hidden_dim,
        )
        self.ffn_norm = RMSNorm(
            hidden_dim,
        )


    def forward(self, x):
        """
        A simple, residual forward 
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.attn_norm(x)) # freqs_cis?
        x = x + self.ffn(self.ffn_norm(x))
        return x