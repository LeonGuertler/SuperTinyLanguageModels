"""
TODO
"""
import torch 
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from models.components.normalization import build_normalization


class Attention(ABC, torch.nn.Module):
    """
    Abstract base class for all attention mechanisms.
    Defines the interface that all attention mechanisms must implement.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        bias: bool = False,
        dropout_p: float = 0.0,
        context_window: int = 2048,
        is_causal: bool = True,
        normalization_name: str = "none",
    ):
        """
        Initialize the Attention module.

        Args:
            hidden_dim (int): The dimensionality of the input embeddings.
            num_q_heads (int): Number of query heads.
            num_kv_heads (int): Number of key/value heads.
            bias (bool, optional): If True, includes a bias term in linear projections. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            context_window (int, optional): Maximum sequence length for positional encodings. Defaults to 512.
            is_causal (bool, optional): If True, applies causal masking. Defaults to True.
        """
        super().__init__()
        assert hidden_dim % num_kv_heads == 0, "Hidden dim must be divisible by num heads"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

        self.is_causal = is_causal
        self.dropout_p = dropout_p
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.group_hidden_dim = hidden_dim // self.num_q_heads * self.num_kv_heads

        # key, query, value projections for all heads
        self.c_attn = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim + 2 * self.group_hidden_dim,
            bias=bias
        )

        # output projection
        self.c_proj = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=bias
        )

        self.normalization = build_normalization(
            normalization_name=normalization_name,
            dim=hidden_dim,
            bias=bias
        )

    def forward(
        self, 
        x: torch.tensor, 
        attn_mask: Optional[torch.tensor] = None
    ):
        """
        Forward pass for the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).
            attn_mask (Optional[torch.Tensor], optional): Attention mask of shape (B, num_heads, S, S). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, S, H).
        """
        # normalize x 
        x = self.normalization(x)

        B, S, H = x.size()
        
        # calculate query, key, values for all heads in batch
        # move head forward to the batch dim 
        q, k, v = self.c_attn(x).split([H, self.group_hidden_dim, self.group_hidden_dim], dim=-1)

        k = k.reshape(B, self.num_kv_heads, S, self.group_hidden_dim//self.num_kv_heads)
        q = q.reshape(B, self.num_q_heads, S, self.group_hidden_dim//self.num_kv_heads)
        v = v.reshape(B, self.num_kv_heads, S, self.group_hidden_dim//self.num_kv_heads)

        # reshape to have same dim as q
        k = k.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1)

        # reshape attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

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









