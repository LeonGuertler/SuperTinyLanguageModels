"""
Shared components of the byte level models.
"""

import torch

from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from models.components.normalization import build_normalization


from models.components.activations import build_activation
from models.components.attention import build_attention
from models.components.normalization import build_normalization

from models.components.positional_encoding import (
    LearnedPosEncoding,
    IdentityEncoding,
    SinCosPosEncoding,
    AbsolutePositionalEncoding,
    ALiBiPosEncoding,
    SANDWICHPosEncoding,
    xPOSPosEncoding,
    TransformerXLRelativePosEncoding,
    T5RelativePosEncoding,
    ShawRelativePosEncoding,
    LearnedRelativePosEncoding,
    build_positional_encodings
)

from models.components.utils.attention_utils import (
    apply_attention
)

class AttentionByte(torch.nn.Module):
    """
    Flexible attention module with support for different attention mechanisms
    and positional encodings.
    """

    def __init__(
        self,
        hidden_dim,
        num_q_heads,
        num_kv_heads,
        bias,
        attention_type,
        pos_encoding_cfg,
        context_window,
        is_causal,
    ):
        super().__init__()
        assert hidden_dim % num_kv_heads == 0, "Hidden dim must be divisible by num_kv_heads"
        assert num_kv_heads % num_q_heads == 0, "num_kv_heads must be divisible by num_q_heads"

        group_size = num_kv_heads // num_q_heads

        # Key, query, value projections for all heads
        self.c_attn = torch.nn.Linear(
            hidden_dim, hidden_dim + 2 * hidden_dim // group_size, bias=bias
        )

        # Output projection
        self.c_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Attention dropout
        self.attn_dropout = torch.nn.Dropout()

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = group_size
        self.is_causal = is_causal

        # Select attention mechanism
        self.attention_type = attention_type
        self.attn_dropout_p = self.attn_dropout.p

        # Initialize the positional encoding
        self.pos_encoding = build_positional_encodings(
            pos_encoding_cfg=pos_encoding_cfg,
            context_window=context_window,
            hidden_dim=hidden_dim,
            num_heads=num_q_heads
        )

    def forward(self, x, attention_mask=None, past_key_value=None):
        """
        Forward pass of the attention module with KV caching.

        Args:
            x (Tensor): Input tensor of shape (B, S, H)
            attention_mask (Tensor, optional): Boolean mask of shape (B, S, S)
                where True indicates positions to be masked out.
            past_key_value (tuple, optional): Tuple of (past_key, past_value) tensors.
                Each of shape (B, num_kv_heads, S_past, head_dim).

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: Output tensor of shape (B, S, H) and updated (key, value).
        """
        B, S, H = x.size()
        num_grouped_heads = self.num_kv_heads // self.group_size
        group_hidden_dim = H // self.group_size

        # Apply absolute positional encoding to input embeddings if applicable
        if isinstance(self.pos_encoding, (LearnedPosEncoding, SinCosPosEncoding, AbsolutePositionalEncoding)):
            x = self.pos_encoding(x)  # x is now positionally encoded

        # Compute query, key, values
        qkv = self.c_attn(x)  # (B, S, H + 2 * H / group_size)
        q, k, v = qkv.split([H, group_hidden_dim, group_hidden_dim], dim=-1)
        q = q.view(B, S, self.num_q_heads, H // self.num_q_heads)
        k = k.view(B, S, num_grouped_heads, H // self.num_kv_heads)
        v = v.view(B, S, num_grouped_heads, H // self.num_kv_heads)

        # If using RoPE or similar, apply after projection
        if isinstance(self.pos_encoding, xPOSPosEncoding):
            q, k = self.pos_encoding(q, k)

        # Transpose for multi-head attention
        q = q.transpose(1, 2)  # (B, num_q_heads, S, head_dim)
        k = k.transpose(1, 2)  # (B, num_kv_heads, S, head_dim)
        v = v.transpose(1, 2)  # (B, num_kv_heads, S, head_dim)

        # Reshape k and v to match q's number of heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Handle KV caching
        if past_key_value is not None:
            # Unpack past keys and values
            past_k, past_v = past_key_value  # Each of shape (B, num_kv_heads, S_past, head_dim)

            # Concatenate past keys and values with current keys and values
            k = torch.cat([past_k, k], dim=2)  # (B, num_kv_heads, S_past + S, head_dim)
            v = torch.cat([past_v, v], dim=2)  # (B, num_kv_heads, S_past + S, head_dim)
        else:
            # No past keys and values
            past_k, past_v = None, None

        # Update past_key_value for next step
        new_past_key_value = (k.detach(), v.detach())

        # Apply attention mechanism
        y = apply_attention(
            attention_mechanism_type=self.attention_type,
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            is_causal=self.is_causal,
            dropout_p=self.attn_dropout_p
        )  # (B, num_q_heads, S, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, S, H)  # Re-assemble all head outputs

        # Output projection
        y = self.attn_dropout(self.c_proj(y))

        return y, new_past_key_value


class GenericTransformerBlockByte(torch.nn.Module):
    """
    A simple transformer block that combines
    FFN, Attn and normalization.
    """

    def __init__(self, hidden_dim, context_window, ffn_cfg, attn_cfg):
        super().__init__()

        # build the attn norm
        self.attn_norm = build_normalization(
            normalization_name=attn_cfg["normalization"],
            dim=hidden_dim,
            bias=attn_cfg["bias"],
        )

        # build the attention
        self.attn = build_attention(
            hidden_dim=hidden_dim,
            context_window=context_window,
            attn_cfg=attn_cfg,
        )
        self.attn = AttentionByte(
            hidden_dim=hidden_dim,
            num_q_heads=8,
            num_kv_heads=8,
            bias=False,
            attention_type="standard",
            pos_encoding_cfg={"positional_encoding_type": "none"},
            context_window=context_window,
            is_causal=True,
        )

        # build the ffn norm
        self.ffn_norm = build_normalization(
            normalization_name=ffn_cfg.get("normalization", "rms_norm"),  # Default: rms_norm
            dim=hidden_dim,
            bias=ffn_cfg["bias"],
        )

        # build the ffn block
        self.ffn = build_ffn(
            hidden_dim=hidden_dim,
            ffn_cfg=ffn_cfg,
        )

    def forward(self, x, attention_mask=None, past_key_value=None):
        """
        Forward pass through the transformer block with KV caching.

        Args:
            x: Input tensor of shape (B, S, H)
            attention_mask: Attention mask tensor
            past_key_value: Tuple of past key and value tensors for KV caching

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: Output tensor and updated past_key_value
        """
        # Attention layer
        attn_norm_x = self.attn_norm(x)
        attn_output, new_past_key_value = self.attn(
            attn_norm_x,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        x = x + attn_output

        # Feed-forward layer
        ffn_norm_x = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_norm_x)
        x = x + ffn_output

        return x, new_past_key_value


class ProjectingFFN(torch.nn.Module):
    """
    A simple feedforward network
    """

    def __init__(
        self,
        hidden_dim,
        output_dim,
        ffn_dim,
        bias,
        ffn_activation,
    ):
        super().__init__()
        # build the ffn block
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.activation = build_activation(activation_name=ffn_activation)

        self.linear_2 = torch.nn.Linear(ffn_dim, output_dim, bias=bias)

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class ByteLevelTransformerBlock(torch.nn.Module):
    """
    A simple transformer block that combines
    FFN, Attn and normalization.
    """

    def __init__(self, input_dim, output_dim, ffn_dim, context_window, use_rope=False):
        super().__init__()

        # build the attn norm
        self.attn_norm = build_normalization(
            normalization_name="rms_norm", dim=input_dim, bias=False
        )

        # build the attention
        self.attn = build_attention(
            hidden_dim=input_dim,
            context_window=context_window,
            attn_cfg={
                "num_q_heads":8,
                "num_kv_heads":8,
                "bias":False,
                "attention_mechanism":"block_sparse",
                "is_causal":False,
                "pos_enc_cfg":{
                    "positional_encoding_type":"rope"
                }
            }
        )

        # build the ffn norm
        self.ffn_norm = build_normalization(
            normalization_name="rms_norm", dim=input_dim, bias=False
        )

        # build the ffn block
        self.ffn = ProjectingFFN(
            hidden_dim=input_dim,
            ffn_dim=ffn_dim,
            output_dim=output_dim,
            bias=False,
            ffn_activation="gelu",
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
            attention_mask: the attention mask
        Returns:
            x: the output tensor (b, s, h)
        """
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = self.ffn(self.ffn_norm(x))
        return x


