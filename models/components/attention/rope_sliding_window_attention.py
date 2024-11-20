import torch
from typing import Optional, Tuple
from models.components.attention import RoPEAttention
from models.components.normalization import build_normalization


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RopeSlidingWindowAttention(RoPEAttention):
    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        window_size: int,
        bias: bool = False,
        dropout_p: float = 0.0,
        context_window: int = 2048,
        is_causal: bool = True,
        normalization_name: str = "none",
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            dropout_p=dropout_p,
            context_window=context_window,
            is_causal=is_causal,
            normalization_name=normalization_name
        )
        self.window_size = window_size
        self.scale = (hidden_dim // num_q_heads) ** -0.5

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with fixed dimension handling."""
        x = self.normalization(x)
        B, S, H = x.size()  # batch, sequence length, hidden dim
        
        # Project to queries, keys, values
        q, k, v = self.c_attn(x).split([H, self.group_hidden_dim, self.group_hidden_dim], dim=-1)
        
        # Print shapes for debugging
        head_dim = H // self.num_q_heads
        
        # Reshape maintaining proper dimensions
        q = q.view(B, S, self.num_q_heads, head_dim)
        k = k.view(B, S, self.num_kv_heads, head_dim)
        v = v.view(B, S, self.num_kv_heads, head_dim)
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:S])
        
        # Repeat for multi-query attention
        k = repeat_kv(k, self.num_q_heads // self.num_kv_heads)
        v = repeat_kv(v, self.num_q_heads // self.num_kv_heads)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [B, num_q_heads, S, head_dim]
        k = k.transpose(1, 2)  # [B, num_q_heads, S, head_dim]
        v = v.transpose(1, 2)  # [B, num_q_heads, S, head_dim]
        
        # Create sliding window attention mask
        window_mask = torch.ones((S, S), dtype=torch.bool, device=x.device)
        for i in range(S):
            start = max(0, i - self.window_size)
            end = min(S, i + self.window_size + 1)
            window_mask[i, start:end] = False
        
        if self.is_causal:
            causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=x.device), diagonal=1)
            window_mask = window_mask | causal_mask
        
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
        
        # Compute attention scores with proper dimensions
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_q_heads, S, S]
        
        # Apply masks
        attn_weights = attn_weights.masked_fill(window_mask, float('-inf'))
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        # Apply attention to values
        y = torch.matmul(attn_weights, v)  # [B, num_q_heads, S, head_dim]
        
        # Reshape output with correct dimensions
        y = y.transpose(1, 2)  # [B, S, num_q_heads, head_dim]
        y = y.reshape(B, S, H)  # [B, S, hidden_dim]
        
        # Final projection
        y = self.c_proj(y)
        
        return y