from models.core_models import GenericTransformer
import torch
from torch import nn
# params: k_interior_layers, lora_rank

class LoRA(nn.Module):
    def __init__(self, linear_layer, lora_rank):
        """Wraps the linear layer with LoRA"""
        super().__init__()
        self.linear_layer = linear_layer
        self.lora_rank = lora_rank
        self.U = nn.Parameter(torch.randn(linear_layer.out_features, lora_rank))
        self.V = nn.Parameter(torch.randn(lora_rank, linear_layer.in_features))
        self.bias = linear_layer.bias
        self.weight = linear_layer.weight

    def forward(self, x):
        """Forward pass through the linear layer with LoRA"""
        # compute the LoRA weight matrix
        W = self.weight + torch.matmul(self.U, self.V)
        return torch.nn.functional.linear(x, W, self.bias)

class FFNInteriorLora(GenericTransformer):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.k_interior_layers = model_cfg["k_interior_layers"]
        self.lora_rank = model_cfg["lora_rank"]
        # share the weights between transformer blocks in layers 1+k_interior_layers to D-k_interior_layers
        base_layer = 1 + self.k_interior_layers
        for i in range(1 + self.k_interior_layers, len(self.transformer.h) - self.k_interior_layers):
            for name, param in self.transformer.h[i].ffn.named_parameters():
                # set the param to the corresponding param in the base layer
                base_param = getattr(self.transformer.h[base_layer].ffn, name)
                param.data = base_param.data
                # wrap the linear layer with LoRA
                if self.lora_rank is not None:
                    setattr(self.transformer.h[i].ffn, name, LoRA(base_param, self.lora_rank))



                
        

                                                                             


# class GenericTransformer(torch.nn.Module):
#     """
#     Generic Transformer Class intended to be used for as
#     broad a range of transformer models as possible.
#     """

#     def __init__(self, model_cfg):
#         super().__init__()

#         # build the transformer
#         self.transformer = torch.nn.ModuleDict(
#             {
#                 "drop": torch.nn.Dropout(),
#                 "h": torch.nn.ModuleList(
#                     [
#                         GenericTransformerBlock(
#                             hidden_dim=model_cfg["hidden_dim"],
#                             context_window=model_cfg["context_window"],
#                             use_rope=model_cfg["positional_encoding_type"] == "rope",
#                             ffn_cfg=model_cfg["core_model"]["ffn"],
#                             attn_cfg=model_cfg["core_model"]["attn"],
#                         )
#                         for _ in range(model_cfg["core_model"]["num_layers"])
#                     ]
#                 ),
#             }
#         )

#     def forward(self, x):
#         """
#         Pass an input through the model
#         Args:
#             x: torch.tensor(B, S, H)
#         Returns:
#             x: torch.tensor(B, S, H)
#         """

#         # apply dropout
#         x = self.transformer.drop(x)

#         # pass through the transformer blocks
#         for block in self.transformer.h:
#             x = block(x)

#         return x