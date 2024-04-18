"""
A collection of basic model building blocks.
"""
import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.utils import apply_rotary_emb


class RMSNorm(nn.Module):
    """RMSNORM"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return input / torch.sqrt(
            torch.mean(input ** 2, dim=-1, keepdim=True) + 1e-8
        ) * self.weight + 0 if self.bias is None else self.bias

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """
    Basic Self-Attention module.
    """
    def __init__(self, hidden_dim, num_heads, bias=False, dropout=0.0, is_causal=False, apply_rope=False):
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

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.is_causal = is_causal
        self.apply_rope = apply_rope

    def forward(self, x, attention_mask=None, rope_freqs=None):
        """
        Forward pass
        """
        assert attention_mask is None, "Not implemented yet"
        B, S, H = (
            x.size()
        ) # batch, sequence, hidden

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.hidden_dim, dim=2)

        # apply rotary embeddings
        if self.apply_rope:
            assert rope_freqs is not None, "rope_freqs must be provided"
            q, k = apply_rotary_emb(q, k, rope_freqs)

        # split heads

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
            is_causal=self.is_causal,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, S, H)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.dropout_layer(self.c_proj(y)) # is this really necessary?
    
        return y

class CrossAttention(nn.Module):
    """
    Basic Cross-Attention module.
    """
    def __init__(self, hidden_dim, num_heads, bias=False, dropout=0.0, apply_rope=False):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads"

        # key, query, value projections for all heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim, bias=bias)

        # output projection
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # regularization
        self.dropout_layer = nn.Dropout(dropout)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.apply_rope = apply_rope

    def forward(self, query, key_value, attention_mask=None, rope_freqs=None):
        """
        Forward pass
        """
        B, S, H = query.size()  # batch, query sequence length, hidden
        _, T, _ = key_value.size()  # batch, key/value sequence length, hidden

        # separate projection for query and combined projection for keys and values
        q = self.q_proj(query)
        k, v = self.kv_proj(key_value).split(self.hidden_dim, dim=2)

        # apply rotary embeddings
        if self.apply_rope:
            assert rope_freqs is not None, "rope_freqs must be provided"
            q, k = apply_rotary_emb(q, k, rope_freqs)

        # reshape and permute the projections to bring the head to the front
        q = q.view(B, S, self.num_heads, H // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)

        # scale queries
        scale = (H // self.num_heads) ** -0.5
        q = q * scale

        # attention mechanism (softmax(QK^T / sqrt(dk))V)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)

        # weighted sum of values
        y = torch.matmul(attn_probs, v)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, S, H)

        # output projection
        y = self.c_proj(y)

        return y

class SwiGLUFNN(nn.Module):
    """SwiGLU from https://arxiv.org/pdf/2002.05202v1.pdf"""
    def __init__(self, hidden_dim, ffn_dim, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc_w = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.swish = nn.SiLU()
        self.c_fc_v = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ffn_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        """
        Forward pass
        """

        w = self.c_fc_w(x)
        w = self.swish(w)
        v = self.c_fc_v(x)
        wv = w * v # hadamard product
        out = self.c_proj(wv)
        out = self.dropout(out)
        return out

class FFN(nn.Module):
    """
    A simple Feed Forward Network block.
    """
    def __init__(self, hidden_dim, ffn_dim, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=bias,
        )

        self.gelu = nn.GELU()
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


class StandardBlock(nn.Module):
    """
    A simple abstraction to combine the 
    LayerNorms, SelfAttention and FeedForward layers
    """
    def __init__(self, hidden_dim, ffn_dim, bias, num_heads, dropout):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            is_causal=True
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = FFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            dropout=dropout,
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

class ModernBlock(nn.Module):
    """Uses RMSNorm, SwiGLUFNN, RoPE..."""
    def __init__(self, hidden_dim, ffn_dim, bias, num_heads, dropout=0.0, apply_rope=True):
        super().__init__()

        self.ln_1 = RMSNorm(hidden_dim, bias=bias)
        self.attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            is_causal=True,
            apply_rope=apply_rope
        )
        self.ln_2 = RMSNorm(hidden_dim, bias=bias)
        self.mlp = SwiGLUFNN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x, attention_mask=None, rope_freqs=None):
        """
        A simple, residual forward 
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.ln_1(x), attention_mask, rope_freqs)
        x = x + self.mlp(self.ln_2(x))
        return x


    
class NextTokenHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.ln = LayerNorm(hidden_dim, bias=True)
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.ln(x)
        logits = self.linear(x)
        return logits
    
class NextSequenceHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.ln = LayerNorm(hidden_dim, bias=True)
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def forward(self, x):
        x = self.ln(x)
        logits = self.linear(x)
        return logits
    

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, rank=32, alpha=1):
        super(LoraLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Original weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # LORA specific parameters
        self.A = nn.Parameter(torch.Tensor(out_features, rank))
        self.B = nn.Parameter(torch.Tensor(rank, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # LORA adaptation
        lora_adaptation = self.alpha * (self.A @ self.B)
        adapted_weight = self.weight + lora_adaptation

        return nn.functional.linear(input, adapted_weight, self.bias)