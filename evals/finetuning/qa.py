"""Training a QA model with finetuning on """

import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from evals.evaluator_interface import EvaluationInterface
from evals.mcqs import load_benchmarks
from models.model_shell import ModelShell


def form_prompt(question, all_options):
    """Form a prompt for the model"""
    option_list = "\n".join(
        [f"{i+1}. {option}" for i, option in enumerate(all_options)]
    )
    return f"{question}\n{option_list}\nAnswer: "


class FinetuningQA(EvaluationInterface):
    """Evaluator class for training and evaluating models.
    Currently just for GLUE finetuning."""

    def __init__(self, model, benchmarks, max_train_samples=1000, max_eval_samples=1000):
        super().__init__(model)
        self.model: ModelShell = model
        self.train_datasets = {}
        self.eval_datasets = {}
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.load_dataset(benchmarks)

    def load_dataset(self, qa_benchmarks):
        """Load the dataset"""
        for benchmark in qa_benchmarks:
            train_dataset = load_benchmarks.load_benchmark(benchmark, split="train")
            eval_dataset = load_benchmarks.load_benchmark(benchmark, split="validation")
            self.train_datasets[benchmark] = train_dataset
            self.eval_datasets[benchmark] = eval_dataset

    def evaluate(self):
        """Evaluate the base model on the dataset"""
        results = {}
        for benchmark in self.train_datasets:
            model = self.train(benchmark)
            results[benchmark] = self.evaluate_model(model, benchmark)
        return results

    def train(self, benchmark):
        """Train SKLearn model on the dataset"""
        train_features, train_labels = self.get_features_labels(benchmark, "train", self.max_train_samples)
        model = LogisticRegression(max_iter=1000)
        model.fit(train_features, train_labels)
        return model

    def evaluate_model(self, model, benchmark):
        """Evaluate sklearn model"""
        print(f"Evaluating Benchmark: {benchmark}")
        eval_features, eval_labels = self.get_features_labels(benchmark, "validation", self.max_eval_samples)
        predictions = model.predict(eval_features)
        accuracy = accuracy_score(eval_labels, predictions)
        return accuracy

    def get_features_labels(self, benchmark, split, max_samples):
        """Extract all features and labels from the dataset"""
        benchmark_features = []
        benchmark_labels = []

        dataset = self.train_datasets if split == "train" else self.eval_datasets

        for i, sample in enumerate(dataset[benchmark]):
            if i >= max_samples:
                break
            features, label = self.extract_features(self.embedding_func, *sample)
            features = features.cpu().numpy()
            benchmark_features.append(features)
            benchmark_labels.append(label)
        return benchmark_features, benchmark_labels

    @torch.no_grad()
    def embedding_func(self, input_str):
        """Embed the input string"""
        init_embeds = self.model.embedding_model.inference(input_str)
        embeds = self.model.core_model(init_embeds)[0]
        embeds = embeds[-1]  # Take last one
        return embeds

    def extract_features(self, embedding_func, question, answer, other_options):
        """Extract features from a sample"""
        placement = random.randint(0, len(other_options))
        options = other_options[:placement] + [answer] + other_options[placement:]
        prompt = form_prompt(question, options)
        embeddings = embedding_func(prompt)
        label = placement
        return embeddings, label
