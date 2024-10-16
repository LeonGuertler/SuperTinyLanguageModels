"""
TODO
"""
import torch 
import math 
from typing import Optional


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerAttention(torch.nn.Module):
    """
    Implements the Linformer self-attention mechanism.
    Projects keys and values to a lower-dimensional space to reduce computational complexity.
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
        k_reduction_factor: int = 4,
        share_kv_proj: bool = False,
    ):
        """
        Initialize the LinformerAttention module.

        Args:
            hidden_dim (int): The dimensionality of the input embeddings.
            num_q_heads (int): Number of query heads.
            num_kv_heads (int): Number of key/value heads.
            bias (bool, optional): If True, includes a bias term in linear projections. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            context_window (int, optional): Maximum sequence length. Defaults to 512.
            is_causal (bool, optional): If True, applies causal masking. Defaults to True.
            k_reduction_factor (int, optional): Factor by which to reduce the key/value dimensions. Defaults to 4.
        """
        super().__init__()

        # Ensure num_kv_heads==num_q_heads
        assert num_kv_heads==num_q_heads, "num_kv_heads must equal num_q_heads for linformer"

        self.context_window = context_window 
        self.num_heads = num_kv_heads
        self.dim_heads = hidden_dim//self.num_heads 

        self.to_q = torch.nn.Linear(hidden_dim, self.dim_heads*self.num_heads, bias=False)

        self.k = hidden_dim // k_reduction_factor
        # projection weights
        self.to_k = torch.nn.Linear(hidden_dim, self.num_heads*self.dim_heads, bias=bias)
        self.proj_k = torch.nn.Parameter(init_(torch.zeros(context_window, self.k)))

        self.share_kv_proj = share_kv_proj 
        if not self.share_kv_proj:
            self.to_v = torch.nn.Linear(hidden_dim, self.dim_heads, bias=bias)
            self.proj_v = torch.nn.Parameter(init_(torch.zeros(context_window, self.k)))

        self.dropout = torch.nn.Dropout(dropout_p)
        self.c_proj = torch.nn.Linear(self.dim_heads*self.num_heads, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None):
        """ TODO """ 
        b, n, d, d_h, h, k = *x.shape, self.dim_heads, self.num_heads, self.k

        kv_len = n #if context is None else context.shape[1]
        assert kv_len <= self.context_window, f'the sequence length of the key / values must be {self.context_window} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x #if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv_proj else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv_proj else self.proj_k)

        # allow for variable sequence lengths (less than maximum sequence length) by slicing projections

        if kv_len < self.context_window:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.c_proj(out)

