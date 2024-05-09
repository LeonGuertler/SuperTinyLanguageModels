"""
A collection of FFN blocks
"""
import torch 

from models.components.layers.activations import build_activation

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
        ):
        super().__init__()
        # build the ffn block
        self.linear_1 = torch.nn.Linear(
            hidden_dim, 
            ffn_dim,
            bias=bias
        )

        self.activation = build_activation(
            activation_name=ffn_activation
        )

        self.linear_2 = torch.nn.Linear(
            ffn_dim, 
            hidden_dim,
            bias=bias
        )

    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x 
    

FFN_DICT = {
    "generic": lambda hidden_dim, ffn_cfg: GenericFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        use_bias=ffn_cfg["use_bias"],
        ffn_activation=ffn_cfg["activation"]
    ),
}

def build_ffn(hidden_dim, ffn_cfg):
    """
    Build a feedforward network
    """
    return FFN_DICT[ffn_cfg["ffn_type"]](
        hidden_dim=hidden_dim,
        ffn_cfg=ffn_cfg
    )