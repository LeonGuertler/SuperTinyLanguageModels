"""
A collection of different model heads.
"""
import torch 


from models.components.layers.normalization import build_normalization


class AutoregressiveLMHead(torch.nn.Module):
    """
    Generic autoregressive language model head.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.layer_norm = build_normalization(
            normalization_name=model_cfg["model_head_norm"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"]
        )
        self.lm_head = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["vocab_size"],
            bias=model_cfg["lm_head_bias"]
        )


