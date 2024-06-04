"""
A collection of metrics for evaluating models
"""
import torch

def accuracy_metric(confidences):
    """
    Calculate the accuracy of the model over a path_prob
    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Outputs:
        accuracy: float of accuracy
    """
    _, predicted = torch.max(confidences, 1)
    return (predicted == 0).float().mean()

def path_confidence(confidences):
    """
    Calculate the path confidence of the model.
    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Returns:
        path_confidence: float of path confidences
    """
    softmaxed = torch.nn.functional.softmax(confidences, dim=-1)
    softmaxed = softmaxed[:, 0]
    return softmaxed.mean()

MCQ_METRIC_DICT = {"accuracy": accuracy_metric,
                   "path_confidence": path_confidence}

