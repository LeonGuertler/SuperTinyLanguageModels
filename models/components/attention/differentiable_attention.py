""" 
TODO 
https://arxiv.org/abs/2410.05258
https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py
 """

import math
from models.components.attention import Attention 
import torch 
import torch.nn.functional as F
from torch import nn
from typing import Optional, Dict, Tuple

from models.components.normalization import RMSNorm

def lambda_init_fn(depth: int) -> float:
    """
    Computes the initial lambda value based on depth.

    Args:
        depth (int): Depth or layer index.

    Returns:
        float: Initialized lambda value.
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key/value tensors along the head dimension.

    Args:
        x (torch.Tensor): Tensor of shape (B, num_kv_heads, S, head_dim).
        n_rep (int): Number of repetitions.

    Returns:
        torch.Tensor: Repeated tensor of shape (B, num_kv_heads * n_rep, S, head_dim).
    """
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class DifferentiableAttention(Attention):
    """
    Implements a Multihead Differentiable Attention mechanism.
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
        depth: int = 1  # New parameter specific to MultiheadDiffAttn
    ):
        """
        Initialize the MultiheadDiffAttn module.

        Args:
            hidden_dim (int): Dimensionality of input embeddings.
            num_q_heads (int): Number of query heads.
            num_kv_heads (int): Number of key/value heads.
            bias (bool, optional): If True, includes bias in projections. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            context_window (int, optional): Maximum sequence length. Defaults to 512.
            is_causal (bool, optional): If True, applies causal masking. Defaults to True.
            depth (int, optional): Depth or layer index for lambda initialization. Defaults to 1.
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
        self.depth = depth
        self.num_heads = num_kv_heads
        self.head_dim = hidden_dim // self.num_heads
        # Initialize lambda parameters
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        # Sub-layer normalization
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for MultiheadDiffAttn.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).
            attn_mask (Optional[torch.Tensor], optional): Attention mask of shape (B, num_heads, S, S). Defaults to None.
            rel_pos (Optional[torch.Tensor], optional): Relative positional embeddings. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, S, H).
        """
        B, S, H = x.size()
        src_len = S

        # Project inputs to Q, K, V
        q, k, v = self.c_attn(x).split([H, self.group_hidden_dim, self.group_hidden_dim], dim=-1)
        
        # Reshape Q, K, V
        # q: (B, S, 2*num_heads, head_dim)
        q = q.view(B, S, 2 * self.num_heads, self.head_dim)
        # k: (B, S, 2*num_kv_heads, head_dim)
        k = k.view(B, src_len, 2 * self.num_kv_heads, self.head_dim)
        # v: (B, S, num_kv_heads, 2*head_dim)
        v = v.view(B, src_len, self.num_kv_heads, 2 * self.head_dim)

        # Apply Rotary Positional Embeddings if provided
        if rel_pos is not None:
            q, k = apply_rotary_emb(q, k, rel_pos, interleaved=True)

        # Adjust dimensions
        offset = src_len - S
        q = q.transpose(1, 2)  # (B, 2*num_heads, S, head_dim)
        k = repeat_kv(k.transpose(1, 2), self.group_size)  # (B, 2*num_kv_heads * group_size, S, head_dim)
        v = repeat_kv(v.transpose(1, 2), self.group_size)  # (B, num_kv_heads * group_size, S, 2*head_dim)

        # Scale Q
        q = q * self.scaling

        # Compute raw attention weights
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # (B, 2*num_heads, S, S)

        # Apply attention mask
        if attn_mask is None:
            if self.is_causal:
                attn_mask = torch.triu(
                    torch.full((S, src_len), float("-inf"), device=x.device),
                    diagonal=1 + offset
                )
            else:
                attn_mask = torch.zeros((S, src_len), device=x.device)
        else:
            attn_mask = attn_mask.view(B, self.num_heads, S, src_len)
        
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask.unsqueeze(1)  # Broadcast mask across grouped heads
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        # Compute lambda terms
        # lambda_1 and lambda_2 shape: (B, 2*num_heads, 1, 1)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).unsqueeze(-1).unsqueeze(-1).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).unsqueeze(-1).unsqueeze(-1).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init  # Shape: (B, 2*num_heads, 1, 1)

        # Reshape attention weights for lambda application
        attn_weights = attn_weights.view(B, self.num_heads, 2, S, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]  # (B, num_heads, S, S)

        # Compute attention output
        attn = torch.matmul(attn_weights, v)  # (B, num_heads, S, 2*head_dim)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.num_heads * 2 * self.head_dim)  # (B, S, H)

        # Final output projection
        attn = self.c_proj(attn)  # (B, S, H)
        return attn


def compute_freqs_cis(seq_len: int, head_dim: int) -> torch.Tensor:
    """
    Computes complex frequencies used for rotary positional encodings.

    Args:
        seq_len (int): Sequence length.
        head_dim (int): Dimension of each head.

    Returns:
        torch.Tensor: Frequencies in complex form, shape (seq_len, head_dim/2).
    """
    freqs = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (seq_len, head_dim/2) complex
    return freqs_cis

def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshapes freqs_cis for broadcasting with input tensor x.

    Args:
        freqs_cis (torch.Tensor): Frequencies in complex form, shape (seq_len, head_dim/2).
        x (torch.Tensor): Input tensor to be rotated, shape (..., head_dim).

    Returns:
        torch.Tensor: Reshaped frequencies, ready for broadcasting.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, "Input tensor must have at least 2 dimensions."
    assert freqs_cis.shape == (x.shape[1], x.shape[-1] // 2), f"freqs_cis shape {freqs_cis.shape} does not match required shape {(x.shape[1], x.shape[-1] // 2)}"
    shape = [1] * ndim
    shape[1] = freqs_cis.shape[0]  # Sequence length
    shape[-1] = freqs_cis.shape[1]  # head_dim/2
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotary embedding to the query and key.

    Args:
        xq (torch.Tensor): Query tensor of shape (B, num_heads, S, head_dim).
        xk (torch.Tensor): Key tensor of shape (B, num_heads, S, head_dim).
        freqs_cis (torch.Tensor): Precomputed frequencies, shape (seq_len, head_dim/2).
        interleaved (bool, optional): If True, assumes even dimensions are real and odd are imaginary. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated queries and keys.
    """
    # Split the head_dim into real and imaginary parts
    # Assuming head_dim is even
    xq_ = xq.view(*xq.shape[:-1], -1, 2).contiguous()  # (B, num_heads, S, head_dim/2, 2)
    xk_ = xk.view(*xk.shape[:-1], -1, 2).contiguous()

    # View as complex numbers
    xq_complex = torch.view_as_complex(xq_)  # (B, num_heads, S, head_dim/2)
    xk_complex = torch.view_as_complex(xk_)  # (B, num_heads, S, head_dim/2)

    # Reshape freqs_cis for broadcasting
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_complex)  # (1, seq_len, head_dim/2)

    # Apply rotation
    xq_rotated = xq_complex * freqs_cis  # (B, num_heads, S, head_dim/2)
    xk_rotated = xk_complex * freqs_cis  # (B, num_heads, S, head_dim/2)

    # Convert back to real numbers
    xq_rotated = torch.view_as_real(xq_rotated).flatten(-2)  # (B, num_heads, S, head_dim)
    xk_rotated = torch.view_as_real(xk_rotated).flatten(-2)

    return xq_rotated.type_as(xq), xk_rotated.type_as(xk)