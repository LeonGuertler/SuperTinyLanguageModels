"""
A collection of metrics for evaluating models
"""

def accuracy_metric(predictions, targets):
    """
    Calculate the accuracy of the model
    """
    correct = 0
    total = 0
    for prediction, target in zip(predictions, targets):
        total += 1
        if prediction == target:
            correct += 1
    return correct / total





MCQ_METRIC_DICT = {
    "accuracy": accuracy_metric
}