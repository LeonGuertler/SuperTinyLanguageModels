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
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            ffn_cfg=model_cfg["ffn"],
                            attn_cfg=model_cfg["attn"],
                            depth=i
                        )
                        for i in range(model_cfg["num_layers"])
                    ]
                ),
            }
        )

        if model_cfg.get("ffn_weight_tying", False):  # Default: False
            # Share the weights between all FFN blocks
            ffn_0 = self.transformer.h[0].ffn
            for i in range(1, len(self.transformer.h)):
                ffn_i = self.transformer.h[i].ffn
                for name, parameter in ffn_0.named_parameters():
                    # Access the parameter of the target FFN block
                    target_param = dict(ffn_i.named_parameters())[name]
                    
                    # Share the storage of the parameter using .data
                    target_param.data = parameter.data
                    print(name)

        if model_cfg.get("cproj_weight_tying", False): # Default: False
            # Share the weights between all CProj blocks
            cproj_0 = self.transformer.h[0].attn.c_proj
            for i in range(1, len(self.transformer.h)):
                for name, module in cproj_0.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        target_module = dict(self.transformer.h[i].attn.c_proj.named_modules())[name]
                        target_module.weight = module.weight
                        target_module.bias = module.bias

        if model_cfg.get("attn_weight_tying", False): # Default: False
            # Share the weights between all CProj blocks
            cattn_0 = self.transformer.h[0].attn.c_attn
            for i in range(1, len(self.transformer.h)):
                for name, module in cattn_0.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        target_module = dict(self.transformer.h[i].attn.c_attn.named_modules())[name]
                        target_module.weight = module.weight
                        target_module.bias = module.bias

    def forward(self, x, attn_mask):
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
        for _ in range(8):
            for block in self.transformer.h:
                x = block(x, attn_mask)
            
        return x


