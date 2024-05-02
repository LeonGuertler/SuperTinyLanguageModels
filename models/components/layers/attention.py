"""
A collection of attention layers.
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Basic Self-Attention module.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        bias=False,
        use_rope=False,
        max_context_window=512,
        is_causal=True,
    ):
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
        self.dropout_layer = nn.Dropout()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.is_causal = is_causal
        if self.use_rope:
            assert max_context_window % 2 == 0
            self.freqs_cis = compute_freqs_cis(
                seq_len=max_context_window, head_dim=hidden_dim // num_heads
            ).to(torch.device("cuda"))

    def forward(self, x, attention_mask=None):
        """
        Forward pass
        """
        assert attention_mask is None, "Not implemented yet"
        B, S, H = x.size()  # batch, sequence, hidden

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, S, self.num_heads, H // self.num_heads)  # (B, T, nh, hs)
        q = q.view(B, S, self.num_heads, H // self.num_heads)  # (B, T, nh, hs)
        v = v.view(B, S, self.num_heads, H // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.use_rope:
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis[:S])
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # flash attention
        # pylint: disable=not-callable
        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.dropout_layer.p if self.training else 0,
            is_causal=self.is_causal,
        )
        # pylint: enable=not-callable
        y = (
            y.transpose(1, 2).contiguous().view(B, S, H)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.dropout_layer(self.c_proj(y))  # is this really necessary?

        return y


def _reshape_for_broadcast(freqs_cis, x):
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
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def compute_freqs_cis(seq_len, head_dim):
    """Computes complex frequences used for rotary positional encodings"""
    freqs = 1.0 / (
        10_000 ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )
    t = torch.arange(seq_len * 2, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def build_attention(
    use_rope: bool,
    hidden_dim: int,
    num_heads: int,
    **kwargs,
):
    """
    Build an attention layer
    """
    return SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        use_rope=use_rope,
        **kwargs,
    )
