import torch
from typing import Optional, Tuple
from models.components.attention import Attention, RoPEAttention
from models.components.normalization import build_normalization

# Utility functions (assuming they are in the same module or imported appropriately)
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
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
    """Repeats key/value tensors along the head dimension."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RopeSlidingWindowAttention(RoPEAttention):
    """
    Implements Rotary Positional Embedding (RoPE) with Sliding Window Attention.
    Restricts attention to a local window around each token.
    """

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
        """
        Initialize the RopeSlidingWindowAttention module.

        Args:
            hidden_dim (int): Dimensionality of input embeddings.
            num_q_heads (int): Number of query heads.
            num_kv_heads (int): Number of key/value heads.
            window_size (int): Size of the sliding window.
            bias (bool, optional): If True, includes bias in projections. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            context_window (int, optional): Maximum sequence length for positional encodings. Defaults to 2048.
            is_causal (bool, optional): If True, applies causal masking. Defaults to True.
            normalization_name (str, optional): Name of the normalization layer. Defaults to "none".
        """
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

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for RoPE with Sliding Window Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).
            attn_mask (Optional[torch.Tensor], optional): 
                Existing attention mask of shape (B, num_heads, S, S). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, S, H).
        """
        # Normalize input
        x = self.normalization(x)

        B, S, H = x.size()
        
        # Project to queries, keys, values
        q, k, v = self.c_attn(x).split([H, self.group_hidden_dim, self.group_hidden_dim], dim=-1)
        
        # Reshape for multi-head attention
        k = k.view(B, S, self.num_kv_heads, self.group_hidden_dim // self.num_kv_heads)
        q = q.view(B, S, self.num_q_heads, self.group_hidden_dim // self.num_kv_heads)
        v = v.view(B, S, self.num_kv_heads, self.group_hidden_dim // self.num_kv_heads)

        # Apply Rotary Positional Embedding
        q, k = apply_rotary_emb(
            xq=q, 
            xk=k, 
            freqs_cis=self.freqs_cis[:S]
        )

        # Repeat keys and values for multi-query heads
        k = repeat_kv(k, self.num_q_heads // self.num_kv_heads)
        v = repeat_kv(v, self.num_q_heads // self.num_kv_heads)

        # Transpose for scaled_dot_product_attention
        q = q.transpose(1, 2)  # (B, num_q_heads, S, head_dim)
        k = k.transpose(1, 2)  # (B, num_q_heads, S, head_dim)
        v = v.transpose(1, 2)  # (B, num_q_heads, S, head_dim)

        # Create Sliding Window Mask
        # Each token can attend to tokens within [i - window_size, i + window_size]
        # and apply causal masking if enabled
        device = x.device
        window_size = self.window_size

        # Generate indices for sequence length
        idxs = torch.arange(S, device=device)
        idxs_i = idxs.view(-1, 1)  # (S, 1)
        idxs_j = idxs.view(1, -1)  # (1, S)

        # Compute absolute distance between tokens
        distance = torch.abs(idxs_i - idxs_j)  # (S, S)

        # Sliding window mask: True where distance > window_size
        sliding_mask = distance > window_size  # (S, S)

        # If causal, ensure that tokens can only attend to previous tokens
        if self.is_causal:
            causal_mask = idxs_j > idxs_i  # (S, S)
            combined_mask = sliding_mask | causal_mask
        else:
            combined_mask = sliding_mask  # (S, S)

        # Expand mask to (1, 1, S, S) to broadcast over batch and heads
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # Combine with existing attention mask if provided
        if attn_mask is not None:
            # Ensure attn_mask is of shape (B, num_heads, S, S)
            # Combine masks using logical OR
            combined_mask = combined_mask | attn_mask

        # Apply scaled dot-product attention with the combined mask
        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=combined_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False  # Causality handled in the mask
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, S, H)  # (B, S, H)
        y = self.c_proj(y)  # (B, S, H)

        return y
