import torch
import torch.nn as nn
import math
from models.core_models import GenericTransformer

class LoRA(nn.Module):
    def __init__(self, linear_layer: nn.Linear, lora_rank: int, lora_alpha: float = 1.0):
        """
        Wraps the linear layer with LoRA (Low-Rank Adaptation)
        
        Args:
            linear_layer (nn.Linear): The linear layer to be wrapped
            lora_rank (int): The rank of the LoRA matrices
            lora_alpha (float): Scaling factor for the LoRA update
        """
        super().__init__()
        self.linear_layer = linear_layer
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_rank

        self.lora_A = nn.Parameter(torch.empty((lora_rank, linear_layer.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((linear_layer.out_features, lora_rank)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer with LoRA"""
        return self.linear_layer(x) + (self.lora_B @ self.lora_A @ x.T).T * self.scaling

class SharedInteriorFFNLora(GenericTransformer):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.k_interior_layers = model_cfg["k_interior_layers"]
        self.lora_rank = model_cfg["lora_rank"]
        self.lora_alpha = model_cfg.get("lora_alpha", 1.0)
        
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
                        lora_module = LoRA(module, self.lora_rank, self.lora_alpha)
                        setattr(target_module, name, lora_module)

class SharedInteriorFFNLoraAndCProj(SharedInteriorFFNLora):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        
        # Apply LoRA to c_proj in attention layers
        self._apply_weight_sharing_and_lora(
            start_layer=1 + self.k_interior_layers,
            end_layer=len(self.transformer.h) - self.k_interior_layers,
            module_name='attn.c_proj'
        )