"""
A collection of common activation functions.
"""
import torch

class LearnedActivation(torch.nn.Module):
    def __init__(self, hidden_size=10):
        super(LearnedActivation, self).__init__()
        self.fc1 = torch.nn.Linear(1, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights to the learned parameters
        self.fc1.weight.data = torch.tensor([[-0.3478],
                                             [-0.3444],
                                             [-0.9863],
                                             [-0.8657],
                                             [-0.0148],
                                             [ 0.1085],
                                             [-0.5282],
                                             [-0.1138],
                                             [-1.1070],
                                             [-0.1035]])
        self.fc1.bias.data = torch.tensor([ 1.4480,  1.4610, -0.8526,  0.0151, -0.1249, -0.7658,  2.2386, -0.8884, 1.0032, -0.6235])
        self.fc2.weight.data = torch.tensor([[-0.4762, -1.2194,  0.4155,  0.3927, -0.2778,  0.0986, -0.9284,  0.2070, 0.3586, -0.2143]])
        self.fc2.bias.data = torch.tensor([4.1740])
    
    def forward(self, x):
        # Flatten the input to apply the learned activation element-wise
        orig_shape = x.shape
        x = x.view(-1, 1)  # Flatten to (N, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(orig_shape)  # Reshape back to original shape


class IndividualLearnedActivation(torch.nn.Module):
    def __init__(self, input_size, hidden_size=10):
        super(IndividualLearnedActivation, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.fc1_layers = torch.nn.ModuleList([torch.nn.Linear(1, hidden_size) for _ in range(input_size)])
        self.fc2_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, 1) for _ in range(input_size)])
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for all fc1 and fc2 layers to the learned parameters
        for i in range(self.input_size):
            self.fc1_layers[i].weight.data = torch.tensor([[-0.3478],
                                                           [-0.3444],
                                                           [-0.9863],
                                                           [-0.8657],
                                                           [-0.0148],
                                                           [ 0.1085],
                                                           [-0.5282],
                                                           [-0.1138],
                                                           [-1.1070],
                                                           [-0.1035]])
            self.fc1_layers[i].bias.data = torch.tensor([ 1.4480,  1.4610, -0.8526,  0.0151, -0.1249, -0.7658,  2.2386, -0.8884, 1.0032, -0.6235])
            self.fc2_layers[i].weight.data = torch.tensor([[-0.4762, -1.2194,  0.4155,  0.3927, -0.2778,  0.0986, -0.9284,  0.2070, 0.3586, -0.2143]])
            self.fc2_layers[i].bias.data = torch.tensor([4.1740])

    def forward(self, x):
        # Apply the learned activation function to each neuron separately
        orig_shape = x.shape
        x = x.view(-1, self.input_size)  # Flatten to (N, input_size)
        outputs = []
        for i in range(self.input_size):
            xi = x[:, i].unsqueeze(1)  # Get the i-th input neuron
            xi = torch.relu(self.fc1_layers[i](xi))
            xi = self.fc2_layers[i](xi)
            outputs.append(xi)
        x = torch.cat(outputs, dim=1)
        return x.view(orig_shape)  # Reshape back to original shape

ACTIVATIONS_DICT = {
    "gelu": lambda input_size: torch.nn.GELU(),
    "relu": lambda input_size: torch.nn.ReLU(),
    "leakyrelu": lambda input_size: torch.nn.LeakyReLU(),
    "tanh": lambda input_size: torch.nn.Tanh(),
    "sigmoid": lambda input_size: torch.nn.Sigmoid(),
    "silu": lambda input_size: torch.nn.SiLU(),
    "learned": lambda input_size: LearnedActivation(hidden_size=10),
    "individual_learned": lambda input_size: IndividualLearnedActivation(input_size, hidden_size=10),
    "none": lambda input_size: torch.nn.Identity(),
}


def build_activation(activation_name: str, input_size: int = None):
    """
    Given the name of the activation function,
    build it.
    Args:
        activation_name: str
    Returns:
        activation: torch.nn.Module
    """
    return ACTIVATIONS_DICT[activation_name.lower()](
        input_size=input_size
    )
