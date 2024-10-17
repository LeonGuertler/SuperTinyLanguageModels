""" 
TODO 
https://arxiv.org/abs/2410.05258
https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py
"""
import torch 
import math 
from models.components.attention import Attention
from models.components.normalization import RMSNorm
from typing import Optional

class DifferentiableAttention(torch.nn.Module):
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
        depth: int = 1, # the depth of the current layer
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
            out_features=hidden_dim*2 + 2 * self.group_hidden_dim,
            bias=bias
        )

        # output projection
        self.c_proj = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=bias
        )

        # lambda
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth) # from the paper
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(self.group_hidden_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(self.group_hidden_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(self.group_hidden_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(self.group_hidden_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        # self.norm = RMSNorm(self.group_hidden_dim)

        # Compute frequencies for RoPE and register as buffer
        # buffering is necessary to ensure correct device
        freqs_cis = compute_freqs_cis(
            max_seq_len=context_window,
            head_dim=hidden_dim // num_q_heads
        )
        self.register_buffer('freqs_cis', freqs_cis)


    def _scaled_dot_product_attention(
        self, 
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor, 
        attn_mask: Optional[torch.Tensor], 
        dropout_p: float, 
        is_causal: bool
    ) -> (torch.Tensor, torch.Tensor):
        """
        TODO
        """
        B, nh, S, H = q.size()

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(H)

        if is_causal:
            # Create causal mask
            causal_mask = torch.tril(
                torch.ones((S, S), device=q.device)
            ).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, S, S)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        if attn_mask is not None:
            # Apply provided mask 
            scores = scores.mask_fill(attn_mask==0, float('-inf'))

        # Apply Softmax
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        # extract the negative heads, and combine them to create final attention weights
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init


        attn_weights = attn_weights.view(B, nh//2, 2, S, S)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        # attn = self.norm(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(B, S, -1)

        attn = self.c_proj(attn)
        return attn



    def forward(
        self, 
        x: torch.tensor, 
        attn_mask: Optional[torch.tensor] = None
    ):
        """ TODO """
        B, S, H = x.size()
        
        # calculate query, key, values for all heads in batch
        # move head forward to the batch dim 
        q, k, v = self.c_attn(x).split([H*2, self.group_hidden_dim, self.group_hidden_dim], dim=-1)

        k = k.reshape(B, self.num_kv_heads, S, self.group_hidden_dim//self.num_kv_heads)
        q = q.reshape(B, self.num_q_heads*2, S, self.group_hidden_dim//self.num_kv_heads)
        v = v.reshape(B, self.num_kv_heads, S, self.group_hidden_dim//self.num_kv_heads)

        # reshape to have same dim as q
        k = k.repeat_interleave(self.num_q_heads//self.num_kv_heads*2, dim=1)
        v = v.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1)

        y = self._scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
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