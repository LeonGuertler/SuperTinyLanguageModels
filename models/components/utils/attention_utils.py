"""
A collection of alternatives to the standard scaled_dot_product_attention,
including:
- Sparse attention (mostly for memory and speed gains)
"""

import torch
import math
import torch.nn.functional as F


def sparse_attention(query, key, value, attn_mask=None, is_causal=False, dropout_p=0.0):
    """
    Applies sparse scaled dot-product attention without computing the full attention matrix.
    
    Args:
        query (Tensor): Query tensor of shape (B, H, Q_LEN, D)
        key (Tensor): Key tensor of shape (B, H, K_LEN, D)
        value (Tensor): Value tensor of shape (B, H, K_LEN, D)
        attn_mask (Tensor, optional): Boolean mask of shape (B, H, Q_LEN, K_LEN). 
            True indicates disallowed positions.
        is_causal (bool, optional): If True, applies causal masking. Default: False.
        dropout_p (float, optional): Dropout probability. Default: 0.0.
    
    Returns:
        Tensor: Output tensor of shape (B, H, Q_LEN, D)
    """
    # B, H, Q_LEN, D = query.shape
    # _, _, K_LEN, _ = key.shape
    # device = query.device
    # dtype = query.dtype

    # # Create causal mask if needed
    # if is_causal:
    #     causal_mask = torch.triu(torch.ones(Q_LEN, K_LEN, dtype=torch.bool, device=device), diagonal=1)
    #     causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, Q_LEN, K_LEN)
    #     if attn_mask is None:
    #         attn_mask = causal_mask
    #     else:
    #         attn_mask = attn_mask | causal_mask  # Combine masks
    # elif attn_mask is None:
    #     attn_mask = torch.zeros(B, H, Q_LEN, K_LEN, dtype=torch.bool, device=device)

    B, H, Q_LEN, D = query.shape
    _, _, K_LEN, _ = key.shape
    device = query.device
    dtype = query.dtype

    # Ensure attn_mask has the correct shape
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, K_LEN)
            attn_mask = attn_mask.expand(B, H, Q_LEN, K_LEN)
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, Q_LEN, K_LEN)
            attn_mask = attn_mask.expand(B, H, Q_LEN, K_LEN)
        elif attn_mask.dim() == 4:
            # attn_mask is already (B, H, Q_LEN, K_LEN)
            pass
        else:
            raise ValueError(f"Invalid attn_mask dimensions: {attn_mask.dim()}")
    else:
        attn_mask = torch.zeros(B, H, Q_LEN, K_LEN, dtype=torch.bool, device=device)


    # Invert the mask to get allowed positions
    allowed_positions = ~attn_mask  # Shape: (B, H, Q_LEN, K_LEN)
    
    # Get indices of allowed positions
    batch_idx, head_idx, q_idx, k_idx = allowed_positions.nonzero(as_tuple=True)
    num_entries = batch_idx.size(0)

    if num_entries == 0:
        # No allowed positions; return zero tensor
        return torch.zeros(B, H, Q_LEN, D, dtype=dtype, device=device)

    # Flatten batch and head dimensions for processing
    query_flat = query.reshape(B * H, Q_LEN, D)    # Shape: (B*H, Q_LEN, D)
    key_flat = key.reshape(B * H, K_LEN, D)        # Shape: (B*H, K_LEN, D)
    value_flat = value.reshape(B * H, K_LEN, D)    # Shape: (B*H, K_LEN, D)

    # Map multi-dimensional indices to flat indices
    bh_idx = batch_idx * H + head_idx  # Combined batch and head indices

    # Gather the relevant queries, keys, and values
    query_selected = query_flat[bh_idx, q_idx]      # Shape: (num_entries, D)
    key_selected = key_flat[bh_idx, k_idx]          # Shape: (num_entries, D)
    value_selected = value_flat[bh_idx, k_idx]      # Shape: (num_entries, D)

    # Compute attention scores
    scaling = float(D) ** -0.5
    attn_scores = (query_selected * key_selected).sum(dim=-1) * scaling  # Shape: (num_entries,)

    # Prepare to compute softmax per query position
    combined_q_idx = bh_idx * Q_LEN + q_idx  # Unique index per query position
    unique_q_indices, inverse_indices = combined_q_idx.unique(return_inverse=True)

    # Compute the maximum score per query for numerical stability
    max_scores = torch.zeros_like(unique_q_indices, dtype=dtype, device=device)
    max_scores.index_put_(
        (inverse_indices,),
        attn_scores,
        accumulate=False
    )
    max_scores = max_scores.scatter_reduce(
        0, inverse_indices, attn_scores, reduce="amax"
    )

    # Normalize scores
    attn_scores_exp = torch.exp(attn_scores - max_scores[inverse_indices])

    # Sum of exponentiated scores per query
    sum_exp_scores = torch.zeros_like(unique_q_indices, dtype=dtype, device=device)
    sum_exp_scores.index_put_(
        (inverse_indices,),
        attn_scores_exp,
        accumulate=True
    )

    # Compute attention weights
    attn_weights = attn_scores_exp / sum_exp_scores[inverse_indices]  # Shape: (num_entries,)

    # Apply dropout to attention weights
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=query.requires_grad)

    # Compute weighted values
    weighted_values = value_selected * attn_weights.unsqueeze(-1)  # Shape: (num_entries, D)

    # Initialize output tensor
    output = torch.zeros(B * H * Q_LEN, D, dtype=dtype, device=device)  # Shape: (B*H*Q_LEN, D)

    # Sum weighted values per query
    output.index_put_(
        (combined_q_idx,),
        weighted_values,
        accumulate=True
    )

    # Reshape output to (B, H, Q_LEN, D)
    output = output.view(B, H, Q_LEN, D)

    return output

import torch
import torch.nn.functional as F
def block_sparse_attention(query, key, value, attn_mask, is_causal=False, dropout_p=0.0):
    """
    Applies block-sparse scaled dot-product attention based on the attention mask.

    Args:
        query (Tensor): Query tensor of shape (B, H, S, D)
        key (Tensor): Key tensor of shape (B, H, S, D)
        value (Tensor): Value tensor of shape (B, H, S, D)
        attn_mask (Tensor): Boolean mask of shape (B, S, S). True indicates disallowed positions.
        is_causal (bool, optional): If True, applies causal masking within blocks. Default: False.
        dropout_p (float, optional): Dropout probability. Default: 0.0.

    Returns:
        Tensor: Output tensor of shape (B, H, S, D)
    """
    B, H, S, D = query.shape
    device = query.device
    dtype = query.dtype

    output = torch.zeros_like(query)  # Initialize output tensor

    # Iterate over each sequence in the batch
    for b in range(B):
        # Extract the attention mask for the sequence
        seq_attn_mask = attn_mask[b]  # Shape: (S, S)

        # Identify the blocks along the diagonal
        blocks = []
        s = 0
        while s < S:
            # Find the start of the next block
            while s < S and seq_attn_mask[s, s].item() == True:
                s += 1
            if s >= S:
                break
            # Find the end of the block
            start = s
            while s < S and seq_attn_mask[s, s].item() == False:
                s += 1
            end = s
            blocks.append((start, end))

        # Process each block for all heads simultaneously
        for (start, end) in blocks:
            block_len = end - start
            if block_len <= 0:
                continue

            # Slice queries, keys, values for the block
            q_block = query[b, :, start:end, :]    # Shape: (H, block_len, D)
            k_block = key[b, :, start:end, :]      # Shape: (H, block_len, D)
            v_block = value[b, :, start:end, :]    # Shape: (H, block_len, D)

            # Compute attention scores
            scaling = float(D) ** -0.5
            attn_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scaling  # (H, block_len, block_len)

            # Apply causal mask if needed
            if is_causal:
                causal_mask = torch.triu(torch.ones(block_len, block_len, device=device, dtype=torch.bool), diagonal=1)
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            # Apply attention mask within the block
            block_mask = seq_attn_mask[start:end, start:end]  # Shape: (block_len, block_len)
            attn_scores = attn_scores.masked_fill(block_mask.unsqueeze(0), float('-inf'))

            # Compute attention probabilities
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=dropout_p, training=query.requires_grad)

            # Compute attention output
            out_block = torch.matmul(attn_probs, v_block)  # Shape: (H, block_len, D)

            # Place the output back into the output tensor
            output[b, :, start:end, :] = out_block

    return output

ATTENTION_MECHANISMS = {
    "standard": lambda query, key, value, attn_mask, is_causal, dropout_p: torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout_p=dropout_p
    ),
    "sparse": lambda query, key, value, attn_mask, is_causal, dropout_p: sparse_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout_p=dropout_p
    ),
    "block_sparse": lambda query, key, value, attn_mask, is_causal, dropout_p: block_sparse_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout_p=dropout_p
    )
}


def apply_attention(
    attention_mechanism_type,
    query, 
    key, 
    value, 
    attn_mask=None, 
    is_causal=True, 
    dropout_p=0.0
):
    """Apply the attention mechanism."""
    return ATTENTION_MECHANISMS[attention_mechanism_type](
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout_p=dropout_p
    )
