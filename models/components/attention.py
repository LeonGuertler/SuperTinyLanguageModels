"""
A collection of attention layers.
"""

import torch
from models.components.utils.attention_utils import use_attention_type


class Attention(torch.nn.Module):
    """
    Basic but flexible attention module.
    """

    def __init__(
        self,
        attention_type,
        hidden_dim,
        num_q_heads,
        num_kv_heads,
        bias,
        use_rope,
        context_window,
        is_causal,
    ):
        super().__init__()
        assert hidden_dim % num_kv_heads == 0, "Hidden dim must be divisible by num heads"
        assert num_kv_heads % num_q_heads == 0, "num_kv_heads must be divisible by num_q_heads"

        group_size = num_kv_heads // num_q_heads

        # set the attention_type
        self.attention_type = attention_type

        # key, query, value projections for all heads
        self.c_attn = torch.nn.Linear(
            hidden_dim, hidden_dim + 2 * hidden_dim // group_size, bias=bias
        )

        # output projection
        self.c_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # attention dropout
        self.attn_dropout = torch.nn.Dropout()

        self.num_kv_heads = num_kv_heads
        self.group_size = group_size
        self.is_causal = is_causal

        # rope
        self.use_rope = use_rope
        if self.use_rope:
            assert context_window % 2 == 0
            self.freqs_cis = compute_freqs_cis(
                seq_len=context_window, head_dim=hidden_dim // num_kv_heads
            )

    def forward(self, x, attention_mask=None):
        """
        Forward pass
        """
        assert attention_mask is None, "Not implemented yet"
        B, S, H = x.size()
        num_grouped_heads = self.num_kv_heads // self.group_size
        group_hidden_dim = H // self.group_size

        # calculate query, key, values for all heads in batch
        # move head forward to be the batch dim
        q, k, v = self.c_attn(x).split([H, group_hidden_dim, group_hidden_dim], dim=-1)
        k = k.view(B, S, num_grouped_heads, H // self.num_kv_heads)  # (B, T, nh, hs)
        q = q.view(B, S, self.num_kv_heads, H // self.num_kv_heads)  # (B, T, nh, hs)
        v = v.view(B, S, num_grouped_heads, H // self.num_kv_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.use_rope:
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis[:S].to(x.device))
        q = q.transpose(1, 2)  # (B, nh,  d, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)

        # reshape to have same dim as q
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # flash attention
        # pylint: disable=not-callable
        y, attn_components = use_attention_type(
            attention_type=self.attention_type,
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0,
            is_causal=self.is_causal,
        )

        # store the attention components
        self.attn_components = attn_components

        # pylint: enable=not-callable
        y = (
            y.transpose(1, 2).contiguous().view(B, S, H)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.attn_dropout(self.c_proj(y))  # is this really necessary?

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


ATTENTION_DICT = {
    "causal": lambda attention_type, hidden_dim, context_window, use_rope, attn_cfg: Attention(
        attention_type=attention_type,
        hidden_dim=hidden_dim,
        num_kv_heads=attn_cfg["num_kv_heads"],
        num_q_heads=attn_cfg.get("num_q_heads", attn_cfg["num_kv_heads"]),
        bias=attn_cfg.get("bias", False), # Default to False
        use_rope=use_rope,
        context_window=context_window,
        is_causal=True,
    ),
    "bidirectional": lambda attention_type, hidden_dim, context_window, use_rope, attn_cfg: Attention(
        atttention_type=attention_type,
        hidden_dim=hidden_dim,
        num_kv_heads=attn_cfg["num_kv_heads"],
        num_q_heads=attn_cfg.get("num_q_heads", attn_cfg["num_kv_heads"]),
        bias=attn_cfg.get("bias", False), # Default to False
        use_rope=use_rope,
        context_window=context_window,
        is_causal=False,
    )
}


def build_attention(attention_type, hidden_dim, context_window, use_rope, attn_cfg):
    """
    Build an attention layer

    Args:
        hidden_dim: hidden dimension
        context_window: context window
        use_rope: whether to use rope
        attn_cfg: attention config
    """
    return ATTENTION_DICT[attn_cfg["attn_type"]](
        attention_type=attention_type,
        hidden_dim=hidden_dim,
        context_window=context_window,
        use_rope=use_rope,
        attn_cfg=attn_cfg,
    )
