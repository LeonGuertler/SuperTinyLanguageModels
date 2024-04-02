import torch 
import torch.nn as nn 
import math


def gpt2_weights_init(module, depth):
    """
    Initialize model weights according to GPT-2 defaults.
    This function is adjusted to work when called from an external file.
    The 'depth' parameter is now explicitly passed to the function.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in module.named_parameters():
        if pn.endswith("c_proj.weight"):
            torch.nn.init.normal_(
                p, mean=0.0, std=0.02 / math.sqrt(2 * depth)
            )