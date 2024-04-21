"""
A collection of different FFN blocks
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from models.components.layers.activations import (
    build_activation
)

from models.components.layers.moe import (
    MoE
)

class FFN(nn.Module):
    """
    A simple Feed Forward Network block.
    """
    def __init__(self, hidden_dim, ffn_dim, bias=False, dropout=0.0, ffn_activation:str="gelu"):
        super().__init__()
        self.c_fc = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=bias,
        )

        self.gelu = build_activation(
            activation_name=ffn_activation
        )
        self.c_proj = nn.Linear(
            ffn_dim,
            hidden_dim,
            bias=bias,
        )
        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        """
        Forward pass
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class LLama3FFN(nn.Module):
    """
    Implementation based on:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        multiple_of=256,
    ):
        super().__init__()
        ffn_dim = int(2 * ffn_dim / 3)
        ffn_dim = multiple_of * ((ffn_dim + multiple_of -1) // multiple_of)

        self.lin_1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.lin_2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.lin_3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x):
        """
        Forward pass
        """
        return self.lin_2(
            F.silu(self.lin_1(x)) * self.lin_3(x)
        )



class JetMoEFFN(nn.Module):
    """
    Implementation based on: https://github.com/myshell-ai/JetMoE/blob/main/jetmoe/modeling_jetmoe.py
    """
    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        num_experts,
        top_k,
        bias
    ):
        super().__init__()
        self.mlp = MoE(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias,
        )

    def forward(self, x):
        """ Foward pass """
        x_mlp, mlp_aux_loss = self.mlp(x)
        return x_mlp, mlp_aux_loss