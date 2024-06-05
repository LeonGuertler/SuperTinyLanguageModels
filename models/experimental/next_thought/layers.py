"""
Layers that are specific to the next thought models
"""
import torch 
import math 


class AttentionPoolingRemoval(torch.nn.Module):
    """
    Transformer block that removes the top-k
    least paid-attention to tokens.
    """
    def __init__(self, hidden_size_in, hidden_size_out, num_attention_heads, pct_pool_per_layer):
        super().__init__()
        self.pct_pool = pct_pool_per_layer
        self.hidden_size_in = hidden_size_in

        self.attention = CustomMultiHeadAttention(hidden_size_in, num_attention_heads)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size_in, 4 * hidden_size_in),
            torch.nn.GELU(),
            torch.nn.Linear(4 * hidden_size_in, hidden_size_out),
            torch.nn.Dropout(),
        )

        self.norm1 = torch.nn.LayerNorm(hidden_size_in)
        self.norm2 = torch.nn.LayerNorm(hidden_size_out)
        
    def forward(self, x):
        # Apply multi-head attention
        attn_output, attn_output_weights = self.attention(x, x, x)

        # average the attention weights across heads
        attn_output_weights = attn_output_weights.mean(dim=1)

        # find how much each token was attended to on average
        attn_output_weights = attn_output_weights.mean(dim=-2)


        # Normalize and add residual connection
        x = self.norm1(x + attn_output)

        # Calculate the top-k indices to keep based on the attention scores
        seq_len = x.shape[1]
        top_k = int(seq_len * (1 - self.pct_pool))  # Keeping the top 60%

        # Get the indices for the top-k tokens based on the highest attention scores
        _, top_k_indices = torch.topk(attn_output_weights, top_k, dim=-1)

        # Reshape idx tensor to match weights
        idx_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))

        # Use torch.gather to gather values from weights tensor based on indices
        reduced_x = torch.gather(x, 1, idx_expanded)

        # Apply feedforward network and normalization
        reduced_x = self.norm2(self.ffn(reduced_x))

        return reduced_x

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.
    """
    # Q * K^T
    scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
    
    # Scale by the square root of the key dimension
    d_k = query.size(-1)
    scores = scores / math.sqrt(d_k)

    # Apply mask if provided (optional, for example, in Transformer Decoders)
    #if mask is not None:
    #    scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax to get attention weights
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    # Multiply by the value to get the final attention output
    output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, depth_per_head)
    #input(attention_weights.size())

    return output, attention_weights


class CustomMultiHeadAttention(torch.nn.Module):
    """
    Custom implementation of multi-head attention from scratch.
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be evenly divisible by number of heads."
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth_per_head = hidden_size // num_heads

        # Linear layers for projecting into multiple heads
        self.query_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.key_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.value_proj = torch.nn.Linear(hidden_size, hidden_size)

        # Final linear layer for output projection
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)

    def split_into_heads(self, x):
        """
        Split into multiple heads, reshaping accordingly.
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Reshape and split into heads
        x = x.view(batch_size, seq_len, self.num_heads, self.depth_per_head)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth_per_head)

        return x

    def forward(self, q, k, v):
        """
        x: (batch_size, seq_len, hidden_size)
        """
        # Project into queries, keys, and values
        query = self.split_into_heads(self.query_proj(q))
        key = self.split_into_heads(self.key_proj(k))
        value = self.split_into_heads(self.value_proj(v))

        # Apply scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(query, key, value)

        # Concatenate the heads
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(q.size(0), q.size(1), self.hidden_size)

        # Final projection to maintain consistent output
        output = self.out_proj(attention_output)

        return output, attention_weights



class LatentSpaceDecoder(torch.nn.Module):
    """
    Uses a fixed number of heads to decode 
    the latent space into the same hidden dim 
    as the sequence
    """
    def __init__(self, hidden_dim, decoding_length, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoding_length = decoding_length
        self.latent_dim = latent_dim

        self.decoding_layer = torch.nn.Linear(
            in_features=latent_dim,
            out_features=hidden_dim*decoding_length
        )

    def forward(self, x):
        """
        x: (batch_size, latent_dim)
        """
        # TODO, this only needs to be computed once
        batch_size = x.size(0)

        # Project the latent space into the hidden dimension
        x = self.decoding_layer(x)
        x = x.view(batch_size, self.decoding_length, self.hidden_dim)

        return x
    
class LatentSpaceQuery(torch.nn.Module):
    """
    Lets the decoder query the latent space
    """
    def __init__(self, hidden_dim, latent_decoded_length, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_decoded_length = latent_decoded_length
        self.latent_dim = latent_dim

        # k,v come from latent space
        # q comes from the sequence
        self.attention = CustomMultiHeadAttention(
            hidden_size=hidden_dim,
            num_heads=12
        )

    def forward(self, x, latent_space):
        """
        x: (batch_size, seq_len, hidden_dim)
        latent_space: (batch_size, latent_decoded_length, hidden_dim)
        """

        # Query the latent space
        x, _ = self.attention(
            q=x,
            k=latent_space,
            v=latent_space
        )

        return x