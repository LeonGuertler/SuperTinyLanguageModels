"""
TODO
"""
import torch 
from models.components.attention import Attention
from typing import Optional



class RotaryPEMultiHeadAttention(torch.nn.Module):
    """
    Taken from: https://nn.labml.ai/transformers/rope/index.html
    """
    def __init__(
        self, 
        heads: int, 
        hidden_dim: int, 
        rope_percentage: float=0.5,
        dropout_p: float=0.0
        ):
        # super().
        pass


class RoPE(torch.nn.Module):
    """
    https://arxiv.org/pdf/2104.09864
    """
    pass



class RoPEAttention(Attention):
    """
    Implements Rotary Positional Embedding (RoPE) within the Attention mechanism.
    Applies rotational transformations to queries and keys based on their positions.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        bias: bool = False,
        dropout_p: float = 0.0,
        context_window: int = 512,
        is_causal: bool = True,
    ):
        """
        Initialize the RoPEAttention module.

        Args:
            hidden_dim (int): Dimensionality of input embeddings.
            num_q_heads (int): Number of query heads.
            num_kv_heads (int): Number of key/value heads.
            bias (bool, optional): If True, includes bias in projections. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            context_window (int, optional): Maximum sequence length for positional encodings. Defaults to 512.
            is_causal (bool, optional): If True, applies causal masking. Defaults to True.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            dropout_p=dropout_p,
            context_window=context_window,
            is_causal=is_causal,
        )

        # Compute frequencies for RoPE and register as buffer
        # buffering is necessary to ensure correct device
        freqs_cis = compute_freqs_cis(
            max_seq_len=context_window,
            head_dim=hidden_dim // num_q_heads
        )
        self.register_buffer('freqs_cis', freqs_cis)


    def forward(
        self, 
        x: torch.tensor, 
        attn_mask: Optional[torch.tensor] = None
    ):
        """ TODO """
        B, S, H = x.size()
        
        # calculate query, key, values for all heads in batch
        # move head forward to the batch dim 
        q, k, v = self.c_attn(x).split([H, self.group_hidden_dim, self.group_hidden_dim], dim=-1)
        
        k = k.reshape(B, S, self.num_kv_heads, self.group_hidden_dim//self.num_kv_heads)
        q = q.reshape(B, S, self.num_q_heads, self.group_hidden_dim//self.num_kv_heads)
        v = v.reshape(B, self.num_kv_heads, S, self.group_hidden_dim//self.num_kv_heads)

        # apply rope embedding
        q, k = apply_rotary_emb(
            q=q, 
            k=k, 
            freqs_cis=self.freqs_cis[:S]
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # reshape to have same dim as q
        k = k.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1)

        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
            is_causal=self.is_causal
        )

        # re-assemble all head outputs side by side
        y = y.transpose(1,2).contiguous().view(B, S, H)

        # output projection
        y = self.c_proj(y)

        return y 


def _reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    """
    Apply the rotary embedding to the query and key
    """
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, q_)
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)


def compute_freqs_cis(max_seq_len, head_dim):
    """Computes complex frequences used for rotary positional encodings"""
    freqs = 1.0 / (
        10_000 ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )
    t = torch.arange(max_seq_len * 2, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis