"""
GRIN-Moe
    Arxiv paper: https://arxiv.org/abs/2409.12136
    Implementation based on: https://huggingface.co/microsoft/GRIN-MoE/blob/main/modeling_grinmoe.py
"""

import torch
import torch.nn.functional as F
from models.components.utils.feedforward_utils import sparsemixer

class Expert(torch.nn.Module):
    """
    A simple feedforward network (FFN) as an expert.
    """
    def __init__(self, hidden_dim, ffn_dim, bias):
        super().__init__()
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)  # hidden_dim -> ffn_dim
        self.linear_2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=bias)  # ffn_dim -> hidden_dim

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(x)))  # Apply GELU activation

class GatingNetwork(torch.nn.Module):
    """
    A simple routing function (gating network).
    """
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=-1)  # Apply softmax for gating

class MoEFeedForward(torch.nn.Module):
    """
    Modular MoE feedforward network.
    """
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k, bias):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = torch.nn.ModuleList([Expert(hidden_dim, ffn_dim, bias) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(hidden_dim, num_experts)

    def forward(self, x):
        # Get the output from all experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, S, H)
        gate_outputs = self.gating_network(x)  # (B, S, num_experts)

        # Select the top_k experts
        top_k_gate_values, top_k_gate_indices = torch.topk(gate_outputs, self.top_k, dim=-1)  # (B, S, top_k)

        # Gather the outputs of the top_k experts
        top_k_expert_outputs = torch.gather(
            expert_outputs, 1,
            top_k_gate_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.size(-1))
        )  # (B, S, top_k, H)

        # Weight the top_k expert outputs
        weighted_expert_outputs = (top_k_gate_values.unsqueeze(-1) * top_k_expert_outputs).sum(dim=2)  # (B, S, H)
        return weighted_expert_outputs
    

class GRINMoEBlockSparseTop2MLP(torch.nn.Module):
    def __init__(self, hidden_dim, ffn_dim, bias=False):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim

        self.w1 = torch.nn.Linear(self.hidden_dim, self.ffn_dim, bias=bias)
        self.w2 = torch.nn.Linear(self.ffn_dim, self.hidden_dim, bias=bias)
        self.w3 = torch.nn.Linear(self.hidden_dim, self.ffn_dim, bias=bias)

    def forward(self, hidden_states):
        current_hidden_states = F.gelu(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
    

class GRINMoEFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k, router_jitter_noise, input_jitter_noise, bias):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = torch.nn.ModuleList([
            GRINMoEBlockSparseTop2MLP(hidden_dim, ffn_dim, bias) for _ in range(num_experts)
        ])
        self.gating_network = GatingNetwork(hidden_dim, num_experts)

        ## jitter parameters
        self.router_jitter_noise = router_jitter_noise
        self.input_jitter_noise = input_jitter_noise

    def forward(self, x):
        
        batch_size, sequence_length, hidden_dim = x.shape
        if self.training and self.input_jitter_noise > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise)
        x = x.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # print ( 'moe', self.iter, torch.norm(hidden_states).item())
        router_logits = self.gating_network(x)
        
        routing_weights, selected_experts = sparsemixer(
            router_logits, 
            top_k=self.top_k, 
            jitter_eps=self.router_jitter_noise, 
            training=self.training,
        )

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim) 
        # print ( 'moe', self.iter, torch.norm(final_hidden_states).item())
        return final_hidden_states