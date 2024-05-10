model:
    embedding_model_type: str "generic"
    core_model_type: str "generic"
    model_head_type: str "lm_head"
    model_shell_type: str "standard"
    embedding_weight_tying: bool

    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    context_window: int

    ffn:
        ffn_type: str 
        ffn_dim: int 
        activation: str
        normalization: str 
        bias: bool

    attn:
        attn_type: str "generic"
        num_heads: int
        normalization: str
        group_size: int 
        bias: bool

    model_head_norm: str "layer_norm"
    lm_head_bias: bool





model:
    embedding_model_type: "rope"
    rope_i: 10_000