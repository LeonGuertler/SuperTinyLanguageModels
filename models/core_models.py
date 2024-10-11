"""
Simple, flexible core models.
"""

import torch

from models.components.transformer_blocks import GenericTransformerBlock


class GenericTransformer(torch.nn.Module):
    """
    Generic Transformer Class intended to be used for as
    broad a range of transformer models as possible.
    """

    def __init__(self, model_cfg):
        super().__init__()

        # build the transformer
        self.transformer = torch.nn.ModuleDict(
            {
                "drop": torch.nn.Dropout(),
                "h": torch.nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            attention_type=model_cfg.get("attention_type", "standard"),
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            use_rope=model_cfg["positional_encoding_type"] == "rope",
                            ffn_cfg=model_cfg["ffn"],
                            attn_cfg=model_cfg["attn"],
                        )
                        for _ in range(model_cfg["num_layers"])
                    ]
                ),
            }
        )

        if model_cfg.get("ffn_weight_tying", False): # Default: False
            # Share the weights between all FFN blocks, similar to:
            # https://arxiv.org/abs/2402.16840
            ffn_0 = self.transformer.h[0].ffn
            for i in range(1, len(self.transformer.h)):
                for name, module in ffn_0.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        target_module = dict(self.transformer.h[i].ffn.named_modules())[name]
                        target_module.weight = module.weight
                        target_module.bias = module.bias

        if model_cfg.get("cproj_weight_tying", False): # Default: False
            # Share the weights between all CProj blocks
            cproj_0 = self.transformer.h[0].attn.c_proj
            for i in range(1, len(self.transformer.h)):
                for name, module in cproj_0.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        target_module = dict(self.transformer.h[i].attn.c_proj.named_modules())[name]
                        target_module.weight = module.weight
                        target_module.bias = module.bias

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

        # init matrices to get attention dict
        self.attn_components = []

        # pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
            # append the attention components
            self.attn_components.append(block.attn.attn_components)

        return x


