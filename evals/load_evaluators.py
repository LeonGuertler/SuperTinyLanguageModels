"""
Given an evaluator name, load the evaluator
"""

from evals.mcqs.mcq_evaluator import MCQEvaluator

EVALUATORS_DICT = {"mcq": MCQEvaluator}


def load_evaluator(evaluator_name, model):
    """
    Given the evaluator name, load the evaluator
    """
    return EVALUATORS_DICT[evaluator_name](model)
