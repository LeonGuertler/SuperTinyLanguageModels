"""
Simple LoRA FFN weight sharing but with softmax-weighted experts.
"""
import torch 
import math 

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
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        # Initialize gate linear
        self.gate_linear = torch.nn.Linear(in_features, n_experts)

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