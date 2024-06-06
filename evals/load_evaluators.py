"""
Given an evaluator name, load the evaluator
"""

from evals.evaluator_interface import EvaluationInterface
from evals.mcqs.mcq_evaluator import MCQEvaluator
from evals.finetuning.glue import FinetuningEvaluator

EVALUATORS_DICT = {"mcq": MCQEvaluator, "glue": FinetuningEvaluator}


def load_evaluator(evaluator_name, model) -> EvaluationInterface:
    """
    Given the evaluator name, load the evaluator
    """
    return EVALUATORS_DICT[evaluator_name](model)
