"""
A collection of FFN blocks
"""

import torch
import torch.nn.functional as F

from models.components.activations import build_activation
from models.components.ffn import *
from einops import rearrange


class GenericFFN(torch.nn.Module):
    """
    A simple feedforward network
    """

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        bias,
        ffn_activation,
        ffn_dropout
    ):
        super().__init__()
        # build the ffn block
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.activation = build_activation(activation_name=ffn_activation)

        self.linear_2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=bias)

        self.dropout = torch.nn.Dropout(
            p=ffn_dropout
        )

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class SiluFFN(torch.nn.Module):
    """
    Implementation based on:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    originally from https://arxiv.org/abs/2002.05202

    N.B. does not support dropout
    """

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        bias,
        ffn_dropout
    ):
        super().__init__()
        # build the linear functions
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.linear_2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=bias)

        self.linear_3 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.dropout = torch.nn.Dropout(
            p=ffn_dropout
        )

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.dropout(x)
        return self.linear_2(F.silu(self.linear_1(x)) * self.linear_3(x))


class MixtureOfExpertFFN(torch.nn.Module):
    """
    X is the input tensor of shape (B, S, H)
    """
   
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k, bias):
        super().__init__()
        
        self.moe = MoEFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias
        )

    def forward(self, x):
        """
        A simple forward pass through the MOE
        """
        return self.moe(x)


class GRINMixtureOfExpertFFN(torch.nn.Module):
    """
    Implementation based on: https://huggingface.co/microsoft/GRIN-MoE
    Paper here: https://arxiv.org/abs/2409.12136
    """

    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k, router_jitter_noise, input_jitter_noise, bias):
        super().__init__()
        
        self.grinmoe = GRINMoEFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            router_jitter_noise=router_jitter_noise,
            input_jitter_noise=input_jitter_noise,
            bias=bias
        )

    def forward(self, x):

        return self.grinmoe(x)


class BlockShuffleFFN(torch.nn.Module):
    """
    Implementation based on: https://github.com/CLAIRE-Labo/StructuredFFN/tree/main
    """

    def __init__(self, hidden_dim, ffn_dim, bias, return_bias, config, init_config, device="cuda"):
        super().__init__()
        self.device = device

        self.fc1 = BlockShuffleLayer(
            in_features=hidden_dim,
            out_features=ffn_dim,
            bias=bias,
            return_bias=return_bias,
            config=config,
            init_config=init_config,
            device=device)
        self.fc2 = BlockShuffleLayer(
            in_features=ffn_dim,
            out_features=hidden_dim,
            bias=bias,
            return_bias=return_bias,
            config=config,
            init_config=init_config,
            device=device)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class FastFFN(torch.nn.Module):
    """
    Implementation based on: https://github.com/pbelcak/fastfeedforward/tree/main
    """
    def __init__(self, hidden_dim, leaf_width, depth):
        super().__init__()
        
        self.fastffn = FastFeedForward(
            input_width=hidden_dim,
            leaf_width=leaf_width,
            output_width=hidden_dim,
            depth=depth
        )

    def forward(self, x):

        x = self.fastffn(x)

        return x
    
class KANFFN(torch.nn.Module):
    """
    Implementation based on: https://github.com/KindXiaoming/pykan
    """
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        
        self.kanffn1 = KANLayer(
            in_dim=hidden_dim,
            out_dim=ffn_dim,
        )
        self.kanffn2 = KANLayer(
            in_dim=ffn_dim,
            out_dim=hidden_dim,
        )

    def forward(self, x):
        
        batch_size, sequence_length, hidden_dim = x.shape
        x = rearrange(x, 'b s h -> (b s) h')

        x = self.kanffn1(x)
        x = self.kanffn2(x)

        x = x.view(batch_size, sequence_length, hidden_dim)
        return x


FFN_DICT = {
    "generic": lambda hidden_dim, ffn_cfg: GenericFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        bias=ffn_cfg.get("bias", False), # Default to False
        ffn_activation=ffn_cfg.get("activation", "gelu"), # Default to 'gelu
        ffn_dropout=ffn_cfg.get("ffn_dropout", 0.0) # Default to 0.0
    ),
    "silu_ffn": lambda hidden_dim, ffn_cfg: SiluFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        bias=ffn_cfg.get("bias", False), # Default to False
        ffn_dropout=ffn_cfg.get("ffn_dropout", 0.0) # Default to 0.0
    ),
    "moe_ffn": lambda hidden_dim, ffn_cfg: MixtureOfExpertFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        num_experts=ffn_cfg["num_experts"],
        top_k=ffn_cfg["top_k"],
        bias=ffn_cfg.get("bias", False) # Default to False
    ),
    "grin_moe_ffn": lambda hidden_dim, ffn_cfg: GRINMixtureOfExpertFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        num_experts=ffn_cfg["num_experts"],
        top_k=ffn_cfg["top_k"],
        router_jitter_noise=ffn_cfg.get("router_jitter_noise",0.01),
        input_jitter_noise=ffn_cfg.get("input_jitter_noise",0.01),
        bias=ffn_cfg.get("bias", False) # Default to False
    ),
    "block_shuffle_ffn": lambda hidden_dim, ffn_cfg: BlockShuffleFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        bias=ffn_cfg.get("bias", False), # Default to False
        return_bias=ffn_cfg.get("return_bias", True), # Default to False
        config=ffn_cfg.get("config", {}),
        init_config=ffn_cfg.get("init_config", {}),
        device=ffn_cfg.get("device", "cuda")
    ),
    # TODO - WIP - Need to fix the FastFFN and KANFFN implementations
    # fast_ffn doesn't seem fast for inference. Especially slow at evals. 
    # kan_ffn runs out of memory quickly. Need to investigate.
    "fast_ffn": lambda hidden_dim, ffn_cfg: FastFFN(
        hidden_dim=hidden_dim,
        leaf_width=ffn_cfg["leaf_width"],
        depth=ffn_cfg["depth"]
    ),
    "kan_ffn": lambda hidden_dim, ffn_cfg: KANFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"]
    )
}


def build_ffn(hidden_dim, ffn_cfg):
    """
    Build a feedforward network
    """
    assert ffn_cfg["ffn_type"] in FFN_DICT, \
        f"FFN type {ffn_cfg['ffn_type']} not found. Available types: {FFN_DICT.keys()}"
    
    return FFN_DICT[ffn_cfg["ffn_type"]](hidden_dim=hidden_dim, ffn_cfg=ffn_cfg)
