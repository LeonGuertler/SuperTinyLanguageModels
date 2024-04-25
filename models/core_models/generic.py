"""A Transformer entirely defined through configuration..."""

import torch.nn as nn


from models.components.layers.normalization import build_normalization

from models.components.layers.attention import build_attention

from models.components.layers.feedforward import build_ffn

from models.components.positional_encoding import LearnedPosEncoding


class GenericTransformerBlock(nn.Module):
    """
    A simple abstraction to combine the
    LayerNorms, SelfAttention and FeedForward layers

    N.B. this should not be used for moe models
    """

    def __init__(
        self,
        hidden_dim,
        ffn_type,
        ffn_dim,
        bias,
        num_heads,
        normalization="layernorm",
        attn_type="causal",
        attn_group_size=1,
        ffn_activation=None,
    ):
        super().__init__()
        self.norm_1 = build_normalization(
            normalization,
            hidden_dim,
            bias=bias,
        )
        self.attn = build_attention(
            use_rope=(attn_type == "rope"),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            group_size=attn_group_size
        )
        self.norm_2 = build_normalization(
            normalization,
            hidden_dim,
            bias=bias,
        )
        self.mlp = build_ffn(
            ffn_type=ffn_type,
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


class GenericTransformer(nn.Module):
    """Generic Transformer Class intended to be used for as broad a range of
    transformer models as possible.
    """

    def __init__(self, cfg):
        """
        Initialize the transformer with blocks from the config
        """
        super().__init__()

        self.core_model_cfg = cfg["core_model"]
        self.context_window = cfg["model_shell"]["context_window"]

        if self.core_model_cfg["attn_type"] != "rope":
            # build positional encoding
            self.pos_encoder = LearnedPosEncoding(
                hidden_dim=cfg["core_model"]["hidden_dim"],
                context_window=cfg["model_shell"]["context_window"],
            )
        else:
            self.pos_encoder = None

        # build the transformer
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(),
                h=nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            hidden_dim=self.core_model_cfg["hidden_dim"],
                            ffn_type=self.core_model_cfg["ffn_type"],
                            ffn_dim=self.core_model_cfg["ffn_dim"],
                            ffn_activation=self.core_model_cfg["ffn_activation"],
                            num_heads=self.core_model_cfg["num_heads"],
                            bias=self.core_model_cfg["bias"],
                            normalization=self.core_model_cfg["normalization"],
                            attn_type=self.core_model_cfg["attn_type"],
                            attn_group_size=self.core_model_cfg["attn_group_size"],
                        )
                        for _ in range(self.core_model_cfg["depth"])
                    ]
                ),
            )
        )

    def forward(self, x):
        """
        Pass an input through the model
        """
        if self.pos_encoder is not None:
            x = x + self.pos_encoder(x)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)

        return x
