"""
A collection of attention layers.
"""
import math 
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
    
def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply the rotary embedding to the query and key
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def compute_freqs_cis(seq_len, head_dim):
    freqs = 1.0 / (10_000 ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(seq_len*2, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class RoPESelfAttention(nn.Module):
    """
    Self-Attention module with Rotary Positional Encoding and caching.
    Paper: https://arxiv.org/abs/2104.09864
    Implementation based on: https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    # TODO: this shouldn't have the same number of Q,K,V heads
    def __init__(self, hidden_dim, num_heads, context_window, dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        # Key, query, value projections for all heads, but in a batch
        self.wq = nn.Linear(
            hidden_dim,
            self.num_heads * self.head_dim,
            bias=False
        )

        self.wk = nn.Linear(
            hidden_dim,
            self.num_heads * self.head_dim,
            bias=False
        )

        self.wv = nn.Linear(
            hidden_dim,
            self.num_heads * self.head_dim,
            bias=False
        )

        # Output projection
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            hidden_dim,
            bias=False
        )

        self.freqs_cis = compute_freqs_cis(
            seq_len=context_window,
            head_dim=self.head_dim
        ).to(torch.device("cuda"))

    def forward(self, x):
        """
        Forward pass
        """
        B, S, H = x.shape 
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to (B, S, num_heads, head_dim)
        xq = xq.view(B, S, self.num_heads, self.head_dim)
        xk = xk.view(B, S, self.num_heads, self.head_dim)
        xv = xv.view(B, S, self.num_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis[:S])

        # TODO: when implementing GQA, add the repeat function for kv here 

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(output)
    
