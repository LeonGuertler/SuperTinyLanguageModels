import torch.nn as nn
from models.core_models import GenericTransformer

class LoRA(nn.Module):
    def __init__(self, linear_layer, lora_rank):
        """Wraps the linear layer with LoRA"""
        super().__init__()
        self.linear_layer = linear_layer
        self.lora_rank = lora_rank
        self.U = nn.Linear(linear_layer.in_features, lora_rank)
        self.V = nn.Linear(lora_rank, linear_layer.out_features)

    def forward(self, x):
        """Forward pass through the linear layer with LoRA"""
        return self.linear_layer(x) + self.V(self.U(x))

class SharedInteriorFFNLora(GenericTransformer):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.k_interior_layers = model_cfg["k_interior_layers"]
        self.lora_rank = model_cfg["lora_rank"]
        
        self._apply_weight_sharing_and_lora(
            start_layer=1 + self.k_interior_layers,
            end_layer=len(self.transformer.h) - self.k_interior_layers,
            module_name='ffn'
        )

    def _apply_weight_sharing_and_lora(self, start_layer: int, end_layer: int, module_name: str):
        base_module = getattr(self.transformer.h[start_layer], module_name)
        shared_weights = {name: module.weight for name, module in base_module.named_modules() if isinstance(module, nn.Linear)}
        
        for i in range(start_layer, end_layer):
            target_module = getattr(self.transformer.h[i], module_name)
            for name, module in target_module.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight = shared_weights[name]
                    if self.lora_rank is not None:
                        lora_module = LoRA(module, self.lora_rank)
                        setattr(target_module, name, lora_module)



class SharedInteriorFFNLoraAndCProj(SharedInteriorFFNLora):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        # now strictly share the c_proj weights w/o lora
        for i in range(1 + self.k_interior_layers, len(self.transformer.h) - self.k_interior_layers):
            base_cproj = self.transformer.h[1 + self.k_interior_layers].attn.c_proj
            shared_cproj_weights = {name: module.weight for name, module in base_cproj.named_modules() if isinstance(module, nn.Linear)}
            target_cproj = self.transformer.h[i].attn.c_proj
            for name, module in target_cproj.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight = shared_cproj_weights[name]