"""
TODO
"""
import torch 
from models.components.attention import Attention
from typing import Optional


class PerformerAttention(Attention):
    """
    Implements the Performer attention mechanism using the FAVOR+ approximation for linear-time attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: Optional[int] = None,  # Optional; defaults to num_q_heads * group_size
        bias: bool = False,
        dropout_p: float = 0.0,
        context_window: int = 512,
        is_causal: bool = True,
        nb_features: int = 256  # Number of random features
    ):
        """
        Initialize the PerformerAttention module.

        Args:
            hidden_dim (int): Dimensionality of input embeddings.
            num_q_heads (int): Number of query heads.
            num_kv_heads (Optional[int], optional): Number of key/value heads. Defaults to num_q_heads * group_size.
            bias (bool, optional): If True, includes bias in projections. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            context_window (int, optional): Maximum sequence length. Defaults to 512.
            is_causal (bool, optional): If True, applies causal masking. Defaults to True.
            nb_features (int, optional): Number of random features for FAVOR+. Defaults to 256.
        """
        # If num_kv_heads is not provided, set it to num_q_heads * group_size
        if num_kv_heads is None:
            group_size = 8  # You can adjust this based on your model's requirements
            num_kv_heads = num_q_heads * group_size
        else:
            group_size = num_kv_heads // num_q_heads
            assert num_kv_heads % num_q_heads == 0, "num_kv_heads must be divisible by num_q_heads"

        super().__init__(
            hidden_dim=hidden_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            dropout_p=dropout_p,
            context_window=context_window,
            is_causal=is_causal,
        )

        self.nb_features = nb_features

        # Update group_size and group_hidden_dim based on new num_kv_heads
        self.group_size = self.num_kv_heads // self.num_q_heads  # Should be 8
        self.group_hidden_dim = hidden_dim // self.group_size  # 384 // 8 = 48

        # Initialize random projection matrix for FAVOR+
        # Shape: (nb_features, group_hidden_dim)
        projection_matrix = torch.randn(self.nb_features, self.group_hidden_dim)
        projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=1, keepdim=True)
        self.register_buffer('projection_matrix', projection_matrix)

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for PerformerAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).
            attn_mask (Optional[torch.Tensor], optional): Attention mask of shape (B, num_heads, S, S). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, S, H).
        """
        B, S, H = x.size()

        # Project inputs to Q, K, V using the base class's c_attn
        q, k, v = self.c_attn(x).split([H, self.group_hidden_dim, self.group_hidden_dim], dim=-1)

        # Reshape Q, K, V
        # Q: (B, num_kv_heads, S, head_dim)
        q = q.reshape(B, self.num_kv_heads, S, H // self.num_kv_heads)

        # K and V: (B, num_grouped_heads, S, group_hidden_dim)
        k = k.reshape(B, self.num_grouped_heads, S, self.group_hidden_dim)
        v = v.reshape(B, self.num_grouped_heads, S, self.group_hidden_dim)

        # Repeat K and V by group_size to align with Q
        # K, V: (B, num_kv_heads, S, group_hidden_dim)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Random projection for FAVOR+
        # projection_matrix: (nb_features, group_hidden_dim)
        projection = self.projection_matrix.to(x.device)  # Ensure it's on the same device

        # Compute Q and K projections
        # q_proj, k_proj: (B, num_kv_heads, S, nb_features)
        q_proj = torch.einsum('bhld,fd->bhlf', q, projection)
        k_proj = torch.einsum('bhld,fd->bhlf', k, projection)

        # Apply the phi function: exp(-||x||^2 / 2) * (cos(Wx) + sin(Wx))
        # Shape: (B, num_kv_heads, S, nb_features)
        q_phi = torch.exp(-0.5 * (q ** 2).sum(dim=-1, keepdim=True)) * (torch.cos(q_proj) + torch.sin(q_proj))
        k_phi = torch.exp(-0.5 * (k ** 2).sum(dim=-1, keepdim=True)) * (torch.cos(k_proj) + torch.sin(k_proj))

        # Compute normalization terms
        # q_norm: (B, num_kv_heads, 1, nb_features)
        q_norm = q_phi.sum(dim=2, keepdim=True)

        # Compute cumulative sum for causal masking if required
        if self.is_causal:
            # Cumulative sum of k_phi * v along the sequence dimension
            # Shape: (B, num_kv_heads, S, nb_features, group_hidden_dim)
            k_v = k_phi.unsqueeze(-1) * v.unsqueeze(2)  # Broadcasting multiplication
            k_v_cumsum = torch.cumsum(k_v, dim=2)  # Cumulative sum along the sequence

            # Cumulative sum of k_phi along the sequence dimension
            # Shape: (B, num_kv_heads, S, nb_features)
            k_phi_cumsum = torch.cumsum(k_phi, dim=2)

            # Compute the numerator: phi(Q) * cumulative_sum(phi(K) * V)
            # Shape: (B, num_kv_heads, S, group_hidden_dim)
            numerator = (q_phi.unsqueeze(-1) * k_v_cumsum).sum(dim=3)

            # Compute the denominator: phi(Q) * cumulative_sum(phi(K))
            # Shape: (B, num_kv_heads, S, 1)
            denominator = (q_phi * k_phi_cumsum).sum(dim=3, keepdim=True)

            # Avoid division by zero
            y = numerator / (denominator + 1e-8)  # Shape: (B, num_kv_heads, S, group_hidden_dim)
        else:
            # Standard Performer attention (non-causal)
            # Compute KV = phi(K) @ V^T
            # Shape: (B, num_kv_heads, nb_features, group_hidden_dim)
            KV = torch.einsum('bhlf,bhld->bhfl', k_phi, v)

            # Compute the attention output
            # Shape: (B, num_kv_heads, S, group_hidden_dim)
            y = torch.einsum('bhlf,bhfl->bhld', q_phi, KV) / (q_norm @ k_phi.sum(dim=2, keepdim=True).transpose(-2, -1))

        # Reshape y to (B, S, H)
        y = y.transpose(1, 2).contiguous().view(B, S, H)

        # Apply the output projection
        y = self.c_proj(y)

        return y