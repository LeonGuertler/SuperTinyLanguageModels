"""
TODO
"""
import torch 
from models.components.attention import Attention
from typing import Optional, Tuple




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
        context_window: int = 2048,
        is_causal: bool = True,
        normalization_name: str = "none",
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
            normalization_name=normalization_name
        )

        # Compute frequencies for RoPE and register as buffer
        # buffering is necessary to ensure correct device
        freqs_cis = precompute_freqs_cis(
            dim=hidden_dim//num_q_heads,
            end=context_window*2,
            theta=10_000
        )

        self.register_buffer('freqs_cis', freqs_cis)


    def forward(
        self, 
        x: torch.tensor, 
        attn_mask: Optional[torch.tensor] = None
    ):
        """ TODO """
        # normalize x
        x = self.normalization(x)

        B, S, H = x.size()
        
        # calculate query, key, values for all heads in batch
        # move head forward to the batch dim 
        q, k, v = self.c_attn(x).split([H, self.group_hidden_dim, self.group_hidden_dim], dim=-1)
        
        k = k.reshape(B, S, self.num_kv_heads, self.group_hidden_dim//self.num_kv_heads)
        q = q.reshape(B, S, self.num_q_heads, self.group_hidden_dim//self.num_kv_heads)
        v = v.reshape(B, S, self.num_kv_heads, self.group_hidden_dim//self.num_kv_heads)

        # apply rope embedding
        q, k = apply_rotary_emb(
            xq=q, 
            xk=k, 
            freqs_cis=self.freqs_cis[:S]
        )

        # reshape to have same dim as q
        k = repeat_kv(k, self.num_q_heads//self.num_kv_heads)
        v = repeat_kv(v, self.num_q_heads//self.num_kv_heads)
        
        # k = k.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=2)
        # v = v.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=2)


        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # reshape attn_mask
        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        # print(f"{q.size()=}")
        # print(f"{k.size()=}")
        # print(f"{v.size()=}")
        # print(f"{attn_mask.size()=}")

        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0,
            is_causal=self.is_causal
        )

        # re-assemble all head outputs side by side
        y = y.transpose(1,2).contiguous().view(B, S, H)

        # output projection
        y = self.c_proj(y)

        return y 



# taken from https://github.com/meta-llama/llama3/blob/main/llama/model.py
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
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )