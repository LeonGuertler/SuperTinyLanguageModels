"""
A collection of common activation functions.
"""

import torch

ACTIVATIONS_DICT = {
    "gelu": torch.nn.GELU(),
    "relu": torch.nn.ReLU(),
    "leakyrelu": torch.nn.LeakyReLU(),
    "tanh": torch.nn.Tanh(),
    "sigmoid": torch.nn.Sigmoid(),
    "silu": torch.nn.SiLU(),
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
