"""
Simple LoRA FFN weight sharing but with softmax-weighted experts.
"""
import torch 
import math 
from models.core_models import GenericTransformer

from models.components.layers.attention import build_attention
from models.components.layers.feedforward import build_ffn
from models.components.layers.normalization import build_normalization

from models.components.layers.activations import build_activation

class MoELoRA(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_rank: int,
            n_experts: int,
            lora_alpha: float=1.0
        ):
        """
        LoRA MoE implementation

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            lora_rank (int): The rank of the LoRA matrices.
            n_experts (int): Number of experts.
            lora_alpha (float): Scaling factor for the LoRA update.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.lora_rank = lora_rank 
        self.n_experts = n_experts
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_rank

        # Initialize main weight matrix
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))

        # Initialize gate linear
        self.gate_linear = torch.nn.Linear(in_features, n_experts, bias=False)

        # Initialize LoRA matrices
        self.lora_experts_U = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty((lora_rank, in_features)))
            for _ in range(n_experts)
        ])
        self.lora_experts_V = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((out_features, lora_rank)))
            for _ in range(n_experts)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.gate_linear.reset_parameters()
        for i in range(self.n_experts):
            torch.nn.init.kaiming_uniform_(self.lora_experts_U[i], a=math.sqrt(5))
            # V is already initialized to zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass """
        gate = torch.nn.functional.softmax(self.gate_linear(x), dim=-1)
        
        lora_contribution = torch.zeros_like(self.weight)
        for i in range(self.n_experts):
            expert_contribution = self.lora_experts_V[i] @ self.lora_experts_U[i]
            lora_contribution += gate[:, i].unsqueeze(-1).unsqueeze(-1) * expert_contribution

        effective_weight = self.weight + lora_contribution * self.scaling
        return torch.nn.functional.linear(x, effective_weight, self.bias)
    
class SharedMoEFFN(torch.nn.Module):
    """ """
    def __init__(self, hidden_dim, ffn_dim, lora_rank, n_experts, lora_alpha):
        super().__init__()
        self.linear_1 = MoELoRA(
            in_features=hidden_dim,
            out_features=ffn_dim,
            lora_rank=lora_rank,
            n_experts=n_experts,
            lora_alpha=lora_alpha
        )

        self.linear_2 = MoELoRA(
            in_features=ffn_dim,
            out_features=hidden_dim,
            lora_rank=lora_rank,
            n_experts=n_experts,
            lora_alpha=lora_alpha
        )
        self.linear_3 = MoELoRA(
            in_features=hidden_dim,
            out_features=ffn_dim,
            lora_rank=lora_rank,
            n_experts=n_experts,
            lora_alpha=lora_alpha
        )


    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        return self.linear_2(torch.nn.functional.silu(self.linear_1(x)) * self.linear_3(x))
    

class SharedTransformerBlock(torch.nn.Module):
    """
    LoRA shared transformer block
    """
    def __init__(self, hidden_dim, context_window, use_rope, ffn_cfg, attn_cfg):
        super().__init__()

        # build the attn norm
        self.attn_norm = build_normalization(
            normalization_name=attn_cfg["normalization"],
            dim=hidden_dim,
            bias=attn_cfg["bias"],
        )

        # build the attention
        self.attn = build_attention(
            hidden_dim=hidden_dim,
            context_window=context_window,
            use_rope=use_rope,
            attn_cfg=attn_cfg,
        )

        # build the ffn norm
        self.ffn_norm = build_normalization(
            normalization_name=ffn_cfg["normalization"],
            dim=hidden_dim,
            bias=ffn_cfg["bias"],
        )

        # build the ffn block
        self.ffn = SharedMoEFFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_cfg["ffn_dim"],
            lora_rank=ffn_cfg["lora_rank"],
            n_experts=ffn_cfg["n_experts"],
            lora_alpha=ffn_cfg["lora_alpha"],
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
            attention_mask: the attention mask
        Returns:
            x: the output tensor (b, s, h)
        """
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SharedMoE(torch.nn.Module):
    """
    core model class with shared MoE weights
    """
    def __init__(self, model_cfg):
        super().__init__()

        # build the transformer
        self.transformer = torch.nn.ModuleDict(
            {
                "drop": torch.nn.Dropout(),
                "h": torch.nn.ModuleList(
                    [
                        SharedTransformerBlock(
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            use_rope=model_cfg["positional_encoding_type"] == "rope",
                            ffn_cfg=model_cfg["core_model"]["ffn"],
                            attn_cfg=model_cfg["core_model"]["attn"],
                        )
                        for _ in range(model_cfg["core_model"]["num_layers"])
                    ]
                ),
            }
        )

        # share the weights between all ffn blocks
        ffn_0 = self.transformer.h[0].ffn
        for i in range(1, len(self.transformer.h)):
            self.transformer.h[i].ffn.linear_1.weight = ffn_0.linear_1.weight
            self.transformer.h[i].ffn.linear_2.gate_linear.weight = ffn_0.linear_2.gate_linear.weight
            for j in range(ffn_0.linear_1.n_experts):
                self.transformer.h[i].ffn.linear_1.lora_experts_U[j] = ffn_0.linear_1.lora_experts_U[j]
                self.transformer.h[i].ffn.linear_1.lora_experts_V[j] = ffn_0.linear_1.lora_experts_V[j]

            self.transformer.h[i].ffn.linear_2.weight = ffn_0.linear_2.weight
            self.transformer.h[i].ffn.linear_2.gate_linear.weight = ffn_0.linear_2.gate_linear.weight
            for j in range(ffn_0.linear_2.n_experts):
                self.transformer.h[i].ffn.linear_2.lora_experts_U[j] = ffn_0.linear_2.lora_experts_U[j]
                self.transformer.h[i].ffn.linear_2.lora_experts_V[j] = ffn_0.linear_2.lora_experts_V[j]

            self.transformer.h[i].ffn.linear_3.weight = ffn_0.linear_3.weight
            self.transformer.h[i].ffn.linear_3.gate_linear.weight = ffn_0.linear_3.gate_linear.weight
            for j in range(ffn_0.linear_3.n_experts):
                self.transformer.h[i].ffn.linear_3.lora_experts_U[j] = ffn_0.linear_3.lora_experts_U[j]
                self.transformer.h[i].ffn.linear_3.lora_experts_V[j] = ffn_0.linear_3.lora_experts_V[j]

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

        # pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        return x
