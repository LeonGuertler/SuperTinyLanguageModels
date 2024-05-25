"""
A collection of common activation functions.
"""

import torch

class LearnedActivation(torch.nn.Module):
    def __init__(self, hidden_size=10):
        super(LearnedActivation, self).__init__()
        self.fc1 = torch.nn.Linear(1, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Flatten the input to apply the learned activation element-wise
        orig_shape = x.shape
        x = x.view(-1, 1)  # Flatten to (N, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(orig_shape)  # Reshape back to original shape

ACTIVATIONS_DICT = {
    "gelu": torch.nn.GELU(),
    "relu": torch.nn.ReLU(),
    "leakyrelu": torch.nn.LeakyReLU(),
    "tanh": torch.nn.Tanh(),
    "sigmoid": torch.nn.Sigmoid(),
    "silu": torch.nn.SiLU(),
    "learned": LearnedActivation(hidden_size=10),
    "none": torch.nn.Identity(),
}


def build_activation(activation_name: str):
    """
    Given the name of the activation function,
    build it.
    Args:
        activation_name: str
    Returns:
        activation: torch.nn.Module
    """
    return ACTIVATIONS_DICT[activation_name.lower()]
