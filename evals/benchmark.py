import abc
import numpy as np
from sklearn import metrics


class Benchmark(metaclass=abc.ABCMeta):
    """Abstract class for benchmarks."""

    @abc.abstractmethod
    def __init__(self, name, model, description=""):
        self.name = name
        self.description = description
        self.model = model
        self.metrics = {}

    @abc.abstractmethod
    def execute(self):
        """Run the benchmark."""

    def report(self):
        """Print the results of the benchmark."""
        results = {}
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric.aggregate()
        return results


class Metric(metaclass=abc.ABCMeta):
    """Abstract class for metrics."""

    @abc.abstractmethod
    def accumulate(self, prediction, target):
        """Add a sample to the metric."""

    @abc.abstractmethod
    def aggregate(self):
        """Compute the metric and return the result."""

    def batched_accumulate(self, predictions,targets):
        """accumulate but batched"""
        for prediction, target in zip(predictions, targets):
            self.accumulate(prediction, target)


class AccuracyMetric(Metric):
    """Accumulate"""

    def __init__(self):
        self.correct = 0
        self.total = 0

    def accumulate(self, prediction, target):
        self.total += 1
        if prediction == target:
            self.correct += 1

    def aggregate(self):
        return self.correct / self.total


class F1Metric(Metric):
    """Accumulate"""

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def accumulate(self, prediction, target, positive_label="A"):
        if prediction == positive_label and target == positive_label:
            self.tp += 1
        elif prediction == positive_label and target != positive_label:
            self.fp += 1
        elif prediction != positive_label and target == positive_label:
            self.fn += 1

    def aggregate(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)


class FauxModel:
    """Pass"""

    def predict(self, batch, options=None):
        """Pass"""
        if options:
            if type(options) == list and type(options[0]) == list:
                return [options[i][0] for i in range(len(batch))]
            elif type(options) == list:
                return [options[0]] * len(batch)
        else:
            return ["A"]*len(batch)

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        return np.random.rand(len(sentences), 768)
