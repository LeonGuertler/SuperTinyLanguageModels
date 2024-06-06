"""GLUE eval?"""
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

import torch
import numpy as np

from models.model_shell import ModelShell
from evals.evaluator_interface import EvaluationInterface

GLUE_SUBSETS = [
    "cola",
    # "mnli",
    "mrpc",
    # "qnli",
    # "qqp",
    "rte",
    # "sst2",
    "stsb",
    "wnli",
]
GLUE_SENTENCE_MAPPINGS = {
    "cola": ["sentence", 2],
    "mnli": ["premise", "hypothesis", 3],
    "mrpc": ["sentence1", "sentence2", 2],
    "qnli": ["question", "sentence", 2],
    "qqp": ["question1", "question2", 2],
    "rte": ["sentence1", "sentence2", 2],
    "sst2": ["sentence", 2],
    "stsb": ["sentence1", "sentence2", 1],
    "wnli": ["sentence1", "sentence2", 2],
}
GLUE_METRIC = {
    "cola": ["matthews_correlation"],
    "sst2": ["accuracy"],
    "mrpc": ["accuracy", "f1"],
    "stsb": ["pearson", "spearmanr"],
    "qqp": ["accuracy", "f1"],
    "mnli": ["accuracy"],
    "qnli": ["accuracy"],
    "rte": ["accuracy"],
    "wnli": ["accuracy"],
}

METRIC_MAP = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "pearson": pearsonr,
    "spearmanr": spearmanr,
    "matthews_correlation": lambda a, b: matthews_corrcoef(b, a),
    # matthews_corrcoef expects the labels to be the second argument
    # for some reason
}


class FinetuningEvaluator(EvaluationInterface):
    """Evaluator class for training and evaluating models.
    Currently just for GLUE finetuning."""

    def __init__(self, model):
        super().__init__(model)
        self.model: ModelShell = model
        self.train_datasets = {}
        self.eval_datasets = {}
        self.load_dataset()

    def load_dataset(self):
        """Load the dataset for finetuning the model"""
        for subset in GLUE_SUBSETS:
            dataset = load_dataset("glue", subset)
            self.train_datasets[subset] = dataset["train"]
            validation_str = "validation_matched" if subset == "mnli" else "validation"
            self.eval_datasets[subset] = dataset[validation_str]

    def evaluate(self, *args, **kwargs):
        """Evaluate the base model on the dataset"""
        results = {}
        for subset in GLUE_SUBSETS:
            model = self.train(subset)
            results[subset] = self.evaluate_model(model, subset)
        return results

    def train(self, subset):
        """Train SKLearn model on the dataset"""
        train_features, train_labels = self.extract_subset(subset, "train")
        if subset == "stsb":
            model = Ridge()
        else:    
            model = LogisticRegression(max_iter=1000)
        model.fit(train_features, train_labels)
        return model

    def evaluate_model(self, model, subset):
        """Evaluate sklearn model"""
        results = {}
        eval_features, eval_labels = self.extract_subset(subset, "validation")

        predictions = model.predict(eval_features)
        for metric in GLUE_METRIC[subset]:
            metric_func = METRIC_MAP[metric]
            if metric == "pearson":
                metric_val = metric_func(eval_labels, predictions)[0]
            elif metric == "spearmanr":
                metric_val = metric_func(eval_labels, predictions)[1]
            else:
                metric_val = metric_func(eval_labels, predictions)
            results[metric] = metric_val
        return results

    def extract_subset(self, subset, split):
        """Extract all features and labels from the dataset"""
        subset_features = []
        subset_labels = []

        dataset = self.train_datasets if split == "train" else self.eval_datasets

        for sample in dataset[subset]:
            sample_struct = GLUE_SENTENCE_MAPPINGS[subset]
            features, label = self.extract_features(
                self.embedding_func, sample_struct, sample
            )
            features = features.cpu().numpy()
            subset_features.append(features)
            subset_labels.append(label)
        return subset_features, subset_labels

    @torch.no_grad()
    def embedding_func(self, input_str):
        """Embed the input string"""
        init_embeds = self.model.embedding_model.inference(input_str)
        embeds = self.model.core_model(init_embeds)
        embeds = embeds.mean(dim=1)  # mean pooling
        return embeds[0]

    def extract_features(self, embedding_func, sample_struct, sample):
        """Extract features from a sample"""
        strs = [sample[sample_struct[i]] for i in range(len(sample_struct) - 1)]
        embeddings = embedding_func("\n".join(strs))
        label = sample["label"]
        return embeddings, label