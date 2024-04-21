"""
A collection of attention layers.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """
    Basic Self-Attention module.
    """
    def __init__(self, hidden_dim, num_heads, bias=False, dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            hidden_dim,
            3 * hidden_dim,
            bias=bias,
        )

        # output projection
        self.c_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
        )

        # regularization
        self.dropout_layer = nn.Dropout(dropout)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x, attention_mask=None):
        """
        Forward pass
        """
        assert attention_mask is None, "Not implemented yet"
        B, S, H = (
            x.size()
        ) # batch, sequence, hidden

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, S, self.num_heads, H // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, S, self.num_heads, H // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, S, self.num_heads, H // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # flash attention
        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, S, H)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.dropout_layer(self.c_proj(y)) # is this really necessary?
    
        return y