"""
Builds the attention functions specified in the config.

According to the Pytorch documentation, the `torch.nn.functional.scaled_dot_product_attention` function
is efficiently implemented with flash attention. Hence, using the standard function is recommended during training. 

However, where users may want the components of the attention values, the `detailed_scaled_dot_product_attention` function
uses the raw calculations to provide intermediate attention values and the query, key, and value matrices. This might be useful
for inferencing. For example, the adaptive sampler in the `generate.py` script uses the detailed attention function to extract
the scaled attention values (before softmax) for the model's attention heads.
"""

import torch
import torch.nn.functional as F
from typing import Callable


def detailed_scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal) -> torch.Tensor:
        """
        Initializes a detailed attention module that grants users the extraction of intermediate attention values and 
        the query, key, and value matrices.

        Args:
            query (torch.Tensor): Query tensor of shape (..., seq_len_q, dim_k)
            key (torch.Tensor): Key tensor of shape (..., seq_len_k, dim_k)
            value (torch.Tensor): Value tensor of shape (..., seq_len_v, dim_v)
            attn_mask (torch.Tensor, optional): Attention mask.
            dropout_p (float, optional): Dropout probability.
            is_causal (bool): If True, applies a causal mask.
        """
        
        # Compute QK^T
        prenorm_scores = torch.matmul(query, key.transpose(-2, -1))
        # Scale by sqrt(dim_k)
        scaling_factor = key.size(-1) ** 0.5
        scaled_scores = prenorm_scores / scaling_factor

        # Apply attention mask if provided
        if attn_mask is not None:
            scaled_scores = scaled_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply causal mask if required
        if is_causal:
            seq_len = query.size(-2)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=query.device)).bool()
            scaled_scores = scaled_scores.masked_fill(~causal_mask, float('-inf'))

        # Compute attention probabilities
        attn_probs = F.softmax(scaled_scores, dim=-1)

        # Apply dropout if specified
        if dropout_p is not None:
            attn_probs = F.dropout(attn_probs, p=dropout_p)

        # Compute the attention output
        output = torch.matmul(attn_probs, value)

        return output, {"query": query, "key": key, "value": value, "scaled_scores": scaled_scores}

def standard_scaled_dot_product_attention(query, key, value, attn_mask = None, dropout_p = None, is_causal = False) -> torch.Tensor:
    """
    Standard scaled dot-product attention using PyTorch's built-in function.

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        attn_mask (torch.Tensor, optional): Attention mask.
        dropout_p (float, optional): Dropout probability.
        is_causal (bool): If True, applies a causal mask.

    Returns:
        torch.Tensor: The attention output.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal
    ), None

ATTENTION_DICT = {
    "standard": lambda query, key, value, attn_mask, dropout_p, is_causal: standard_scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal),
    "standard_detailed": lambda query, key, value, attn_mask, dropout_p, is_causal: detailed_scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)
}

def use_attention_type(attention_type: str, query, key, value, attn_mask, dropout_p, is_causal) -> Callable:
    """
    Returns the attention function corresponding to the specified type.

    Args:
        attention_type (str): The type of attention to use.

    Returns:
        Callable: The attention function.
    """
    return ATTENTION_DICT[attention_type](query, key, value, attn_mask, dropout_p, is_causal)