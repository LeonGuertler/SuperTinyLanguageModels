"""
A collection of common activation functions.
"""

import enum

import torch


class LearnedActivation(torch.nn.Module):
    def __init__(self, hidden_size=10):
        super(LearnedActivation, self).__init__()
        self.fc1 = torch.nn.Linear(1, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights to the learned parameters
        self.fc1.weight.data = torch.tensor(
            [
                [-0.3478],
                [-0.3444],
                [-0.9863],
                [-0.8657],
                [-0.0148],
                [0.1085],
                [-0.5282],
                [-0.1138],
                [-1.1070],
                [-0.1035],
            ]
        )
        self.fc1.bias.data = torch.tensor(
            [
                1.4480,
                1.4610,
                -0.8526,
                0.0151,
                -0.1249,
                -0.7658,
                2.2386,
                -0.8884,
                1.0032,
                -0.6235,
            ]
        )
        self.fc2.weight.data = torch.tensor(
            [
                [
                    -0.4762,
                    -1.2194,
                    0.4155,
                    0.3927,
                    -0.2778,
                    0.0986,
                    -0.9284,
                    0.2070,
                    0.3586,
                    -0.2143,
                ]
            ]
        )
        self.fc2.bias.data = torch.tensor([4.1740])

    def forward(self, x):
        # Flatten the input to apply the learned activation element-wise
        orig_shape = x.shape
        x = x.view(-1, 1)  # Flatten to (N, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(orig_shape)  # Reshape back to original shape


class ActivationType(enum.Enum):
    """
    Enum for the different types of activations
    """

    GELU = "gelu"
    RELU = "relu"
    LEAKYRELU = "leakyrelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SILU = "silu"
    LEARNED = "learned"
    NONE = "none"


def build_activation(activation_name: str):
    """
    Given the name of the activation function,
    build it.
    Args:
        activation_name: str
    Returns:
        activation: torch.nn.Module
    """
    match activation_name:
        case "gelu":
            return torch.nn.GELU()
        case "relu":
            return torch.nn.ReLU()
        case "leakyrelu":
            return torch.nn.LeakyReLU()
        case "tanh":
            return torch.nn.Tanh()
        case "sigmoid":
            return torch.nn.Sigmoid()
        case "silu":
            return torch.nn.SiLU()
        case "learned":
            return LearnedActivation(hidden_size=10)
        case "none":
            return torch.nn.Identity()
        case _:
            raise ValueError("Invalid activation function")
