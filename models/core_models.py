"""
Simple, flexible core models.
"""

import pydantic
import torch

from models.components.layers.transformer_blocks import GenericTransformerBlock


class CoreModelConfig(pydantic.BaseModel):
    """
    Core Model configuration
    """

    core_model_type: str


class GenericCoreModelConfig(CoreModelConfig):
    """
    Generic Core Model configuration
    """

    positional_encoding_type: str
    ffn: dict
    attn: dict
    num_layers: int


class GenericTransformer(torch.nn.Module):
    """
    Generic Transformer Class intended to be used for as
    broad a range of transformer models as possible.
    """

    def __init__(
        self,
        hidden_dim,
        context_window,
        core_model_cfg: GenericCoreModelConfig,
    ):
        super().__init__()

        # build the transformer
        self.transformer = torch.nn.ModuleDict(
            {
                "drop": torch.nn.Dropout(),
                "h": torch.nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            hidden_dim=hidden_dim,
                            context_window=context_window,
                            ffn_cfg=core_model_cfg.ffn,
                            attn_cfg=core_model_cfg.attn,
                        )
                        for _ in range(core_model_cfg.num_layers)
                    ]
                ),
            }
        )

    def forward(self, x):
        """
        Pass an input through the model
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, H)
        """

        # apply dropout
        x = self.transformer.drop(x)

        # pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        return x


class GenericFFNSharedTransfomer(GenericTransformer):
    """
    Generic Transformer Class that shares the weights
    between all FFN blocks (similar to
    https://arxiv.org/abs/2402.16840).
    """

    def __init__(
        self,
        hidden_dim,
        context_window,
        core_model_cfg: CoreModelConfig,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            context_window=context_window,
            core_model_cfg=core_model_cfg,
        )

        # share the weights between transformer blocks
        ffn_0 = self.transformer.h[0].ffn

        for i in range(1, len(self.transformer.h)):
            # find all linear layers in the ffn subnets and tie them to the first layer
            for name, module in ffn_0.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_module = dict(self.transformer.h[i].ffn.named_modules())[
                        name
                    ]
                    target_module.weight = module.weight
                    target_module.bias = module.bias
