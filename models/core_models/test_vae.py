"""
Remove once working.
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F

import math 



class VAEEncoder(torch.nn.Module):
    """
    Accepts an arbitrary length sequence as input,
    uses the QK^T matrix to, at every layer,
    pick the top n-percent of nodes to pool into 
    a single token (the one paying most attention 
    to the other should be pooled into the other token).
    """
    def __init__(self, hidden_size_in=768, latent_size=4800):
        super().__init__()
        hidden_size = hidden_size_in
        hidden_size_ff = 3072
        num_heads = 12

        self.embedding = torch.nn.Embedding(50304, hidden_size)
        self.positional_encoding = LearnedPosEncoding(768, 512)

        self.standard = torch.nn.ModuleList([
            NormalAttentionBlock(hidden_size, hidden_size_ff, num_heads),
            NormalAttentionBlock(hidden_size, hidden_size_ff, num_heads),
        ])
        h1 = 768
        h2 = 1920
        h3 = 1920
        h4 = 1920
        h5 = 4800
        # convert all to int
        #768 1920 4800 4800 30000
        h1, h2, h3, h4, h5 = map(int, [h1, h2, h3, h4, h5])
        print(h1, h2, h3, h4, h5)

        #768 1919.0 4797.0 11992.0 29979.0


        self.pooling_attention = torch.nn.ModuleList([
            AttentionPoolingRemoval(
                hidden_size_in=h1,
                hidden_size_out=h2,
                num_topk_heads=12, 
                num_attention_heads=12,
                pct_pool_per_layer=0.3,
            ),
            AttentionPoolingRemoval(
                hidden_size_in=h2,
                hidden_size_out=h3,
                num_topk_heads=12, 
                num_attention_heads=12,
                pct_pool_per_layer=0.5
            ),
            AttentionPoolingRemoval(
                hidden_size_in=h3,
                hidden_size_out=h4,
                num_topk_heads=12, 
                num_attention_heads=12,
                pct_pool_per_layer=0.6
            ),
            AttentionPoolingRemoval(
                hidden_size_in=h4,
                hidden_size_out=h5,
                num_topk_heads=12, 
                num_attention_heads=12,
                pct_pool_per_layer=0.6
            ),
        ])




    def forward(self, x):
        # embed the input 
        x = self.embedding(x)

        # apply positional encoding 
        x = x + self.positional_encoding(x)


        # first pass through normal attention blocks
        for layer in self.standard:
            x = layer(x)

        # then pass through pooling attention blocks
        for layer in self.pooling_attention:
            x = layer(x)
        # mean pool final representation
        x = x.mean(dim=-2)
        return x


class FFN(nn.Module):
    """
    A simple Feed Forward Network block.
    """

    def __init__(
        self, hidden_dim, ffn_dim, hidden_size_out=None, bias=False, ffn_activation: str = "gelu"
    ):
        super().__init__()
        hidden_size_out = hidden_size_out if hidden_size_out is not None else hidden_dim
        self.c_fc = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=bias,
        )

        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            ffn_dim,
            hidden_size_out,
            bias=bias,
        )
        self.dropout = nn.Dropout()

    def forward(self, x):
        """
        Forward pass
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class NormalAttentionBlock(torch.nn.Module):
    """
    Attention + FFN + Normalization
    """
    def __init__(self, hidden_size, hidden_size_ff, num_heads):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads)
        self.ff = FFN(hidden_size, hidden_size_ff, hidden_size)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, x):
        # attention
        x = self.norm1(x + self.attention(x, x, x)[0])
        # feedforward
        x = self.norm2(x + self.ff(x))
        return x
    
class LearnedPosEncoding(nn.Module):
    """
    Basic learned positional encoding
    """

    def __init__(self, hidden_dim, context_window):
        super().__init__()
        self.pe = nn.Embedding(num_embeddings=context_window, embedding_dim=hidden_dim)

    def forward(self, x):
        """
        Forward pass∏
        """
        # check device

        if len(x.shape) >= 2:
            return self.pe(torch.arange(x.size(1), device=x.device)).unsqueeze(0).to(x.device)

        else:
            return self.pe(torch.arange(x.size(1), device=x.device))
    
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
    attention_weights = F.softmax(scores, dim=-1)

    # Multiply by the value to get the final attention output
    output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, depth_per_head)
    #input(attention_weights.size())

    return output, attention_weights


class CustomMultiHeadAttention(nn.Module):
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
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Final linear layer for output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

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
    

class RMSNorm(nn.Module):
    """
    RMSNorm (https://arxiv.org/abs/1910.07467), implementation from
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMSNorm"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight 
    

class AttentionPoolingRemoval(nn.Module):
    def __init__(self, hidden_size_in, hidden_size_out, num_topk_heads, num_attention_heads, pct_pool_per_layer):
        super().__init__()
        self.pct_pool = pct_pool_per_layer
        self.hidden_size_in = hidden_size_in

        self.attention = CustomMultiHeadAttention(hidden_size_in, num_attention_heads)

        self.ffn = FFN(hidden_size_in, 4*hidden_size_in, hidden_size_out)  # Feedforward network
        self.norm1 = nn.LayerNorm(hidden_size_in)  # Layer normalization
        self.norm2 = nn.LayerNorm(hidden_size_out)


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
    
class BaseTransformerBlock(nn.Module):
    """
    A simple abstraction to combine the
    LayerNorms, SelfAttention and FeedForward layers
    """

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        ffn_activation,
        bias,
        num_heads,
        normalization="layernorm",
    ):
        super().__init__()
        self.norm_1 = RMSNorm(
            hidden_dim,
        )
        self.attn = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
        )
        self.norm_2 = RMSNorm(
            hidden_dim,
        )
        self.mlp = FFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            bias=bias,
            ffn_activation=ffn_activation,
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
        """
        x = x + self.attn(self.norm_1(x), attention_mask)
        x = x + self.mlp(self.norm_2(x))
        return x

class LatentSpaceDecoder(nn.Module):
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

        self.decoding_layer = nn.Linear(
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
    
class LatentSpaceQuery(nn.Module):
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
    

class CausalSelfAttention(nn.Module):
    """
    Basic Self-Attention module.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        bias=False,
        use_rope=False,
        max_context_window=512,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            hidden_dim,
            3 * hidden_dim,
            bias=bias,
        )

        # output projection
        self.c_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
        )

        # regularization
        self.dropout_layer = nn.Dropout()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rope = use_rope


    def forward(self, x, attention_mask=None):
        """
        Forward pass
        """
        assert attention_mask is None, "Not implemented yet"
        B, S, H = x.size()  # batch, sequence, hidden

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, S, self.num_heads, H // self.num_heads)  # (B, T, nh, hs)
        q = q.view(B, S, self.num_heads, H // self.num_heads)  # (B, T, nh, hs)
        v = v.view(B, S, self.num_heads, H // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # flash attention
        # pylint: disable=not-callable
        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.dropout_layer.p if self.training else 0,
            is_causal=True,
        )
        # pylint: enable=not-callable
        y = (
            y.transpose(1, 2).contiguous().view(B, S, H)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.dropout_layer(self.c_proj(y))  # is this really necessary?

        return y
    


class VAEDecoder(nn.Module):
    """
    Given a latent space representation, decode it into a sequence.
    This should be similar to how VLMs work (i.e. have an encoder
    for the latent space and query it at each step to generate the
    next token).
    """
    def __init__(self, latent_dim, hidden_dim, ffn_dim, vocab_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.vocab_size = vocab_size

        self.latent_embedding = LatentSpaceDecoder(
            hidden_dim=hidden_dim,
            decoding_length=16,
            latent_dim=latent_dim
        )

        self.latent_query = LatentSpaceQuery(
            hidden_dim=hidden_dim,
            latent_decoded_length=16,
            latent_dim=latent_dim
        )

        self.text_encoder = nn.ModuleList(
            [
                BaseTransformerBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    ffn_activation="gelu",
                    bias=False,
                    num_heads=12,
                ) for _ in range(3)
            ]
        )

        self.out_decoder = nn.ModuleList(
            [
                BaseTransformerBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    ffn_activation="gelu",
                    bias=False,
                    num_heads=12,
                ) for _ in range(3)
            ]
        )

        self.lm_head = nn.Linear(
            hidden_dim,
            vocab_size,
            bias=False,
        )

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
        )


        self.positional_encoding = LearnedPosEncoding(768, 512)

    


    def forward(self, x, latent_space):
        """
        x: (batch_size, seq_len, hidden_dim)
        latent_space: (batch_size, latent_dim)
        """
        # decode the latent space
        latent_space = self.latent_embedding(latent_space)

        # embed the text and add positional encoding 
        x = self.embedding(x) + self.positional_encoding(x)

       

        # pass through the text encoder
        for layer in self.text_encoder:
            x = layer(x)

         # query the latent space
        x = x + self.latent_query(x, latent_space)
        # concat 
        #x = torch.cat([q, x], dim=1)

        # pass through the decoder
        for layer in self.out_decoder:
            x = layer(x)

        # project to vocab space
        x = self.lm_head(x)

        return x
    

class VAEEncoderDecoder(nn.Module):
    """
    A simple VAE encoder-decoder model.
    """
    def __init__(self, cfg=None, hidden_size_in=768, latent_size=4800, vocab_size=50304):
        super().__init__()
        self.encoder = VAEEncoder(hidden_size_in=hidden_size_in, latent_size=latent_size)
        self.decoder = VAEDecoder(latent_dim=latent_size, hidden_dim=hidden_size_in, ffn_dim=3072, vocab_size=vocab_size)

    def forward(self, x):
        """
        Forward pass
        """

        latent_space = self.encoder(x)
        #print(x.size(), latent_space.size())
        x = self.decoder(x, latent_space)
        return x