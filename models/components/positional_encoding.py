"""
A collection of positional encoding modules.
"""

import torch
import math


class LearnedPosEncoding(torch.nn.Module):
    """
    Basic learned positional encoding
    """

    def __init__(self, hidden_dim, context_window):
        super().__init__()
        self.pe = torch.nn.Embedding(
            num_embeddings=context_window, embedding_dim=hidden_dim
        )

    def forward(self, x):
        """
        Takes the input tensor and returns it positionally encoded.
        Args:
            x: torch.Tensor of shape (B, S, H)
        Returns:
            torch.Tensor of shape (B, S, H)
        """
        if len(x.shape) >= 2:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, S)
            return x + self.pe(positions)
        else:
            positions = torch.arange(x.size(1), device=x.device)
            return x + self.pe(positions)


class IdentityEncoding(torch.nn.Module):
    """
    In case RoPE is used, there is no need for an initial positional encoding.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Returns the input tensor as is.
        """
        return x


class SinCosPosEncoding(torch.nn.Module):
    """SinCos encoding as used in the Vaswani et al. paper."""

    def __init__(self, hidden_dim, context_window):
        """Set up the pe buffer etc."""
        super().__init__()
        pe = torch.zeros(context_window, hidden_dim)
        position = torch.arange(0, context_window, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe has shape (1, S, H)

        self.pe = torch.nn.Parameter(pe)  # hack for distributed data parallel
        self.pe.requires_grad = False

    def forward(self, x):
        """Add the pe to the input tensor."""
        # x of shape (B, S, H)
        return x + self.pe[:, :x.size(1)]


class AbsolutePositionalEncoding(torch.nn.Module):
    """
    Absolute positional encoding using learned embeddings.
    """
    def __init__(self, hidden_dim, context_window):
        super().__init__()
        self.pe = torch.nn.Embedding(num_embeddings=context_window, embedding_dim=hidden_dim)

    def forward(self, x):
        """
        Add absolute positional encoding to the input tensor.
        Args:
            x: torch.Tensor of shape (B, S, H)
        Returns:
            torch.Tensor of shape (B, S, H)
        """
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, S)
        return x + self.pe(positions)


class ALiBiPosEncoding(torch.nn.Module):
    """
    ALiBi positional encoding.
    """
    def __init__(self, hidden_dim, context_window, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        # Precompute the slopes for each head
        self.register_buffer('slopes', self._get_slopes(num_heads))

    def _get_slopes(self, n):
        # ALiBi slopes as per the paper
        def get_slopes_power_of_2(n):
            start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n), dtype=torch.float32)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[:n - closest_power_of_2]
            return torch.tensor(slopes + extra_slopes, dtype=torch.float32)

    def forward(self, q, k):
        """
        ALiBi modifies the attention scores directly in the Attention class.
        This forward pass returns q and k unchanged.
        """
        return q, k

    def get_bias(self, seq_length, device):
        """
        Generate the ALiBi bias tensor.
        Args:
            seq_length: Sequence length
            device: Device to place the tensor on
        Returns:
            bias: torch.Tensor of shape (num_heads, 1, seq_length)
        """
        # Create a range vector
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)  # (1, S)
        # Compute the ALiBi bias
        bias = self.slopes[:, None] * position_ids[:, None, :]
        return bias.unsqueeze(1)  # (num_heads, 1, 1, S)


class SANDWICHPosEncoding(torch.nn.Module):
    """
    SANDWICH positional encoding.
    Placeholder implementation.
    """
    def __init__(self, hidden_dim, context_window):
        super().__init__()
        # Implement the actual SANDWICH encoding logic here
        self.pe = torch.nn.Parameter(torch.zeros(1, context_window, hidden_dim), requires_grad=False)

    def forward(self, x):
        """
        Add SANDWICH positional encoding to the input tensor.
        Args:
            x: torch.Tensor of shape (B, S, H)
        Returns:
            torch.Tensor of shape (B, S, H)
        """
        return x + self.pe[:, :x.size(1), :]


class xPOSPosEncoding(torch.nn.Module):
    """
    xPOS positional encoding.
    Placeholder implementation.
    """
    def __init__(self, hidden_dim, context_window):
        super().__init__()
        # Implement the actual xPOS encoding logic here
        self.pe = torch.nn.Parameter(torch.zeros(1, context_window, hidden_dim), requires_grad=False)

    def forward(self, x):
        """
        Add xPOS positional encoding to the input tensor.
        Args:
            x: torch.Tensor of shape (B, S, H)
        Returns:
            torch.Tensor of shape (B, S, H)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerXLRelativePosEncoding(torch.nn.Module):
    """
    Transformer-XL Relative Positional Encoding.
    """
    def __init__(self, hidden_dim, context_window, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.relative_embeddings = torch.nn.Embedding(2 * context_window - 1, self.head_dim)

    def forward(self, q, k):
        """
        Compute relative positional bias and apply to attention scores.
        Args:
            q: Queries tensor of shape (B, num_heads, S, head_dim)
            k: Keys tensor of shape (B, num_heads, S, head_dim)
        Returns:
            Modified queries and keys.
        """
        # Placeholder: Actual Transformer-XL relative positional encoding logic
        return q, k

    def get_relative_bias(self, seq_length, device):
        """
        Generate relative positional bias.
        Args:
            seq_length: Sequence length
            device: Device to place the tensor on
        Returns:
            bias: torch.Tensor of shape (num_heads, S, S)
        """
        # Implement relative positional bias computation
        return torch.zeros(self.num_heads, seq_length, seq_length, device=device)


class T5RelativePosEncoding(torch.nn.Module):
    """
    T5 Relative Positional Encoding.
    """
    def __init__(self, hidden_dim, context_window, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.relative_embeddings = torch.nn.Embedding(context_window * 2, self.head_dim)

    def forward(self, q, k):
        """
        Compute relative positional bias and apply to attention scores.
        Args:
            q: Queries tensor of shape (B, num_heads, S, head_dim)
            k: Keys tensor of shape (B, num_heads, S, head_dim)
        Returns:
            Modified queries and keys.
        """
        # Placeholder: Actual T5 relative positional encoding logic
        return q, k

    def get_relative_bias(self, seq_length, device):
        """
        Generate relative positional bias.
        Args:
            seq_length: Sequence length
            device: Device to place the tensor on
        Returns:
            bias: torch.Tensor of shape (num_heads, S, S)
        """
        # Implement relative positional bias computation
        return torch.zeros(self.num_heads, seq_length, seq_length, device=device)


class ShawRelativePosEncoding(torch.nn.Module):
    """
    Relative Positional Encoding as per Shaw et al.
    """
    def __init__(self, hidden_dim, context_window, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.relative_embeddings = torch.nn.Embedding(context_window * 2, self.head_dim)

    def forward(self, q, k):
        """
        Compute relative positional bias and apply to attention scores.
        Args:
            q: Queries tensor of shape (B, num_heads, S, head_dim)
            k: Keys tensor of shape (B, num_heads, S, head_dim)
        Returns:
            Modified queries and keys.
        """
        # Placeholder: Actual Shaw et al. relative positional encoding logic
        return q, k

    def get_relative_bias(self, seq_length, device):
        """
        Generate relative positional bias.
        Args:
            seq_length: Sequence length
            device: Device to place the tensor on
        Returns:
            bias: torch.Tensor of shape (num_heads, S, S)
        """
        # Implement relative positional bias computation
        return torch.zeros(self.num_heads, seq_length, seq_length, device=device)


class LearnedRelativePosEncoding(torch.nn.Module):
    """
    Learned Relative Positional Encoding.
    """
    def __init__(self, hidden_dim, context_window, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.relative_embeddings = torch.nn.Embedding(context_window * 2, self.head_dim)

    def forward(self, q, k):
        """
        Compute relative positional bias and apply to attention scores.
        Args:
            q: Queries tensor of shape (B, num_heads, S, head_dim)
            k: Keys tensor of shape (B, num_heads, S, head_dim)
        Returns:
            Modified queries and keys.
        """
        # Placeholder: Actual Learned relative positional encoding logic
        return q, k

    def get_relative_bias(self, seq_length, device):
        """
        Generate relative positional bias.
        Args:
            seq_length: Sequence length
            device: Device to place the tensor on
        Returns:
            bias: torch.Tensor of shape (num_heads, S, S)
        """
        # Implement relative positional bias computation
        return torch.zeros(self.num_heads, seq_length, seq_length, device=device)


POS_ENCODING_DICT = {
    "learned": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: LearnedPosEncoding(
        hidden_dim=hidden_dim, context_window=context_window
    ),
    "rope": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: IdentityEncoding(),
    "none": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: IdentityEncoding(),
    "sincos": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: SinCosPosEncoding(
        hidden_dim=hidden_dim, context_window=context_window
    ),
    "sandwich": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: SANDWICHPosEncoding(
        hidden_dim=hidden_dim, context_window=context_window
    ),
    "alibi": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: ALiBiPosEncoding(
        hidden_dim=hidden_dim, context_window=context_window, num_heads=num_heads
    ),
    "xpos": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: xPOSPosEncoding(
        hidden_dim=hidden_dim, context_window=context_window
    ),
    "absolute": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: AbsolutePositionalEncoding(
        hidden_dim=hidden_dim, context_window=context_window
    ),
    "transformer_xl": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: TransformerXLRelativePosEncoding(
        hidden_dim=hidden_dim, context_window=context_window, num_heads=num_heads
    ),
    "t5": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: T5RelativePosEncoding(
        hidden_dim=hidden_dim, context_window=context_window, num_heads=num_heads
    ),
    "shaw": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: ShawRelativePosEncoding(
        hidden_dim=hidden_dim, context_window=context_window, num_heads=num_heads
    ),
    "learned_relative": lambda pos_encoding_cfg, hidden_dim, context_window, num_heads: LearnedRelativePosEncoding(
        hidden_dim=hidden_dim, context_window=context_window, num_heads=num_heads
    ),
}


def build_positional_encodings(pos_encoding_cfg, context_window, hidden_dim, num_heads):
    """
    Given the positional encoding config, build it.
    Args:
        pos_encoding_cfg: Configuration dictionary for positional encoding.
        context_window: The maximum context window size.
        hidden_dim: Hidden dimension size.
        num_heads: Number of attention heads.
    Returns:
        positional_encodings: An instance of a positional encoding module.
    """
    encoding_type = pos_encoding_cfg.get("positional_encoding_type", "none")
    if encoding_type not in POS_ENCODING_DICT:
        raise ValueError(f"Unknown positional encoding type: {encoding_type}")
    return POS_ENCODING_DICT[encoding_type](
        pos_encoding_cfg=pos_encoding_cfg,
        hidden_dim=hidden_dim,
        context_window=context_window,
        num_heads=num_heads
    )