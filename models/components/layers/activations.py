"""
A collection of common activation functions.
"""
import torch 
import torch.nn as nn 


def build_activation(activation_name:str):
    """
    Given the name of the activation function,
    build it.
    """
    if activation_name.lower()== "GELU".lower():
        return nn.GELU()
    elif activation_name.lower() == "ReLU".lower():
        return nn.ReLU()
    elif activation_name.lower() == "LeakyReLU".lower():
        return nn.LeakyReLU()
    elif activation_name.lower() == "Tanh".lower():
        return nn.Tanh()
    elif activation_name.lower() == "Sigmoid".lower():
        return nn.Sigmoid()
    elif activation_name.lower() == "silu".lower():
        return nn.SiLU()
    else:
        raise NotImplementedError(f"Activation function {activation_name} not implemented.")