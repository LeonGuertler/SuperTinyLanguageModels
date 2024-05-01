"""A normal transformer but skips some layers and has early exits."""

from models.core_models import generic

import torch


class LayerSkipTransformer(generic.GenericTransformerBlock):
    """
    A transformer that skips some layers and has early exits.
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
        ffn_activation=None,
        max_layer_dropout=1.0,
    ):
        super().__init__(
            hidden_dim,
            ffn_type,
            ffn_dim,
            bias,
            num_heads,
            normalization,
            attn_type,
            ffn_activation,
        )

        self.dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout() for _ in range(len(self.layers))]
        )
        self.max_layer_dropout = max_layer_dropout

    def set_dropouts(self, iteration, max_iteration):
        """
        Set the dropout probability for a given layer
        """
        scale_t = torch.exp(iteration * torch.log(2) / max_iteration) - 1
        for i, dropout in enumerate(self.dropouts):
            depth_layerwise = torch.exp(i * torch.log(2) // (len(self.layers) - 1)) - 1
            dropout.p = scale_t * depth_layerwise * self.max_layer_dropout

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward
        """
        if self.pos_encoder is not None:
            x = x + self.pos_encoder(x)
        x = self.transformer.drop(x)
        early_exits = []
        for i, block in enumerate(self.transformer.h):
            layer_dropout_update_mask = torch.ones(x.size(0), device=x.device)
            layer_dropout_update_mask = self.dropouts[i](layer_dropout_update_mask)
            x[layer_dropout_update_mask] = block(x, attention_mask)[
                layer_dropout_update_mask
            ]
            early_exits.append(x)
        return torch.stack(
            early_exits, dim=1
        )  # very hacky way to pass the early exits all through the head... may cause memory issues...
