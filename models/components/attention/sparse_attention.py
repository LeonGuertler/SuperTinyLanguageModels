"""
TODO
"""
import torch 
from models.components.attention import Attention
from typing import Optional

class SparseAttention(Attention):
    """
    Implements Sparse Attention with a sliding window pattern.
    Each token attends to a fixed number of tokens around it.
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
        window_size: int = 64  # Number of tokens to attend to on each side
    ):
        """
        Initialize the SparseAttention module.

        Args:
            hidden_dim (int): Dimensionality of input embeddings.
            num_q_heads (int): Number of query heads.
            num_kv_heads (int): Number of key/value heads.
            bias (bool, optional): If True, includes bias in projections. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            context_window (int, optional): Maximum sequence length. Defaults to 512.
            is_causal (bool, optional): If True, applies causal masking. Defaults to True.
            window_size (int, optional): Number of tokens to attend to on each side. Defaults to 64.
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

        self.window_size = window_size
        self.group_size = num_q_heads // num_kv_heads  # Ensure this is an integer

        # Validation to ensure head consistency
        assert num_q_heads == num_kv_heads * self.group_size, \
            f"num_q_heads ({num_q_heads}) must be equal to num_kv_heads ({num_kv_heads}) multiplied by group_size ({self.group_size})"

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for SparseAttention with sliding window.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).
            attn_mask (Optional[torch.Tensor], optional): Attention mask of shape (B, num_heads, S, S). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, S, H).
        """
        B, S, H = x.size()
        head_dim = H // self.num_q_heads

        # Compute query, key, value projections
        qkv = self.c_attn(x)  # Shape: (B, S, 3*H)
        q, k, v = qkv.split(H, dim=-1)  # Each of shape: (B, S, H)

        # Reshape and transpose for multi-head attention
        q = q.view(B, S, self.num_q_heads, head_dim).transpose(1, 2)  # (B, num_q_heads, S, head_dim)
        k = k.view(B, S, self.num_kv_heads, head_dim)  # (B, S, num_kv_heads, head_dim)
        v = v.view(B, S, self.num_kv_heads, head_dim)  # (B, S, num_kv_heads, head_dim)

        # Repeat key and value tensors to match the number of query heads
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=2)  # (B, S, num_kv_heads * group_size, head_dim)
            v = v.repeat_interleave(self.group_size, dim=2)  # (B, S, num_kv_heads * group_size, head_dim)
        # Now, reshape to (B, num_q_heads, S, head_dim)
        k = k.view(B, S, self.num_q_heads, head_dim).transpose(1, 2)  # (B, num_q_heads, S, head_dim)
        v = v.view(B, S, self.num_q_heads, head_dim).transpose(1, 2)  # (B, num_q_heads, S, head_dim)

        # Initialize output tensor
        attn_output = torch.zeros_like(q)

        # Apply sliding window attention
        for i in range(S):
            start = max(i - self.window_size, 0)
            end = min(i + self.window_size + 1, S)
            q_slice = q[:, :, i:i+1, :]  # (B, num_heads, 1, head_dim)
            k_slice = k[:, :, start:end, :]  # (B, num_heads, window, head_dim)
            v_slice = v[:, :, start:end, :]  # (B, num_heads, window, head_dim)

            # Compute attention scores
            scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, num_heads, 1, window)

            # Apply attention mask if provided
            if attn_mask is not None:
                mask_slice = attn_mask[:, :, i:i+1, start:end]  # (B, num_heads, 1, window)
                scores = scores.masked_fill(mask_slice == 0, float('-inf'))

            # Apply causal masking if required
            if self.is_causal:
                # Ensure that each position can only attend to previous positions
                relative_positions = torch.arange(start, end, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                mask = relative_positions <= i
                scores = scores.masked_fill(~mask, float('-inf'))

            # Compute attention probabilities
            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = torch.dropout(attn_probs, p=self.dropout_p, train=self.training)

            # Compute attention output
            y = torch.matmul(attn_probs, v_slice)  # (B, num_heads, 1, head_dim)
            attn_output[:, :, i:i+1, :] = y

        # Reassemble all head outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, H)  # (B, S, H)

        # Output projection
        attn_output = self.c_proj(attn_output)  # Assuming c_proj is defined in the parent class

        return attn_output