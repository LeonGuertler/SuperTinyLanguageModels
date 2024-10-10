class Attention(torch.nn.Module):
    """
    Basic but flexible attention module.
    """

    def __init__(
        self,
        hidden_dim,
        num_q_heads,
        num_kv_heads,
        bias,
        use_rope,
        context_window,
        is_causal,
    ):
        super().__init__()
        assert hidden_dim % num_kv_heads == 0, "Hidden dim must be divisible by num heads"
        assert num_kv_heads % num_q_heads == 0, "num_kv_heads must be divisible by num_q_heads"

        group_size = num_kv_heads // num_q_heads

        # key, query, value projections for all heads
        self.c_attn = torch.nn.Linear(
            hidden_dim, hidden_dim + 2 * hidden_dim // group_size, bias=bias
        )

        # output projection
        self.c_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # attention dropout
        self.attn_dropout = torch.nn.Dropout()

        self.num_kv_heads = num_kv_heads
        self.group_size = group_size
        self.is_causal = is_causal

        # rope
        self.use_rope = use_rope
        if self.use_rope:
            assert context_window % 2 == 0
            self.freqs_cis = compute_freqs_cis(
                seq_len=context_window, head_dim=hidden_dim // num_kv_heads
            )

    def forward(self, x, attention_mask=None):
        """
        Forward pass
        """
        assert attention_mask is None, "Not implemented yet"
        B, S, H = x.size()
        num_grouped_heads = self.num_kv_heads // self.group_size
        group_hidden_dim = H // self.group_size

        # calculate query, key, values for all heads in batch
        # move head forward to be the batch dim
        q, k, v = self.c_attn(x).split([H, group_hidden_dim, group_hidden_dim], dim=-1)
        k = k.view(B, S, num_grouped_heads, H // self.num_kv_heads)  # (B, T, nh, hs)
        q = q.view(B, S, self.num_kv_heads, H // self.num_kv_heads)  # (B, T, nh, hs)
        v = v.view(B, S, num_grouped_heads, H // self.num_kv_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.use_rope:
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis[:S].to(x.device))
        q = q.transpose(1, 2)  # (B, nh,  d, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)

        # reshape to have same dim as q
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # flash attention
        # pylint: disable=not-callable
        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0,
            is_causal=self.is_causal,
        )
        # pylint: enable=not-callable
        y = (
            y.transpose(1, 2).contiguous().view(B, S, H)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.attn_dropout(self.c_proj(y))  # is this really necessary?

        return y
