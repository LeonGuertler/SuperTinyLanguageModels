"""Defines the EvaluatorInterface class."""


class EvaluationInterface:
    """Interface for evaluating a model."""

    def __init__(self, model):
        pass

    def evaluate(self):
        """Evaluate the model performance on a list
        of benchmarks."""
        raise NotImplementedError()
