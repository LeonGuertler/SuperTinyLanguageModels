"""Defines the EvaluatorInterface class."""


class EvaluationInterface:
    """Interface for evaluating a model."""

    def __init__(self, model):
        pass

    def evaluate(self, benchmark_names: list[str]):
        """Evaluate the model performance on a list
        of benchmarks."""
        raise NotImplementedError()
