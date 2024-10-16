"""
TODO
"""
import torch 
import math 
from models.components.attention import Attention
from typing import Optional

class ALiBiAttention(Attention):
    """
    Implements ALiBi (Attention with Linear Biases) for causal attention.
    Adds a linear bias to the attention scores based on the relative positions.
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
        Initialize the ALiBiAttention module.

        Args:
            hidden_dim (int): Dimensionality of input embeddings.
            num_q_heads (int): Number of query heads.
            num_kv_heads (int): Number of key/value heads.
            bias (bool, optional): If True, includes bias in projections. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            context_window (int, optional): Maximum sequence length. Defaults to 512.
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
        assert num_kv_heads==num_q_heads, "For ALiBiAttention num_kv_heads has to be equal to num_q_heads"

        self.slopes = self._get_alibi_slopes(self.num_q_heads)

    def _get_alibi_slopes(self, num_heads):
        """
        Compute ALiBi slopes for each head as per the ALiBi paper.

        Args:
            num_heads (int): Number of query heads.

        Returns:
            torch.Tensor: Slopes for each head, shape (num_heads, 1, 1).
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        def get_slopes(n):
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
                return slopes + extra_slopes[: n - closest_power_of_2]

        slopes = get_slopes(num_heads)
        return torch.tensor(slopes).unsqueeze(-1).unsqueeze(-1)  # Shape: (num_heads, 1, 1)

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S, H = x.size()

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split([H, H, H], dim=-1)  # Assuming H divisible by num_heads

        # Reshape and transpose for multi-head attention
        q = q.view(B, S, self.num_q_heads, H // self.num_q_heads).transpose(1, 2)  # (B, num_heads, S, head_dim)
        k = k.view(B, S, self.num_kv_heads, H // self.num_kv_heads).transpose(1, 2)  # (B, num_heads, S, head_dim)
        v = v.view(B, S, self.num_kv_heads, H // self.num_kv_heads).transpose(1, 2)  # (B, num_heads, S, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # (B, num_heads, S, S)


        # ALiBi bias
        alibi_bias = self.slopes.to(x.device) * torch.arange(S, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, num_heads, 1, S)
        attn_scores = attn_scores - alibi_bias  # Apply negative bias

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        if self.is_causal:
            causal_mask = torch.tril(torch.ones((S, S), device=x.device)).bool()
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        # Attention probabilities
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = torch.nn.functional.dropout(attn_probs, p=self.dropout_p, training=self.training)

        # Attention output
        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, S, head_dim)

        # Re-assemble all head outputs side by side
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, H)  # (B, S, H)

        # Output projection
        attn_output = self.c_proj(attn_output)

        return attn_output