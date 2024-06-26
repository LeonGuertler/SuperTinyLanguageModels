"""
Given an evaluator name, load the evaluator
"""

from evals.evaluator_interface import EvaluationInterface
from evals.mcqs.mcq_evaluator import MCQEvaluator
from evals.finetuning.glue import FinetuningEvaluator
from evals.finetuning.qa import FinetuningQA
from evals.mcqs.benchmarks.stories_progression import ProgressionEvaluator

EVALUATORS_DICT = {
    "mcq": MCQEvaluator,
    "glue": FinetuningEvaluator,
    "ft_qa": FinetuningQA,
    "prog": ProgressionEvaluator
}


def load_evaluator(evaluator_name, model, **kwargs) -> EvaluationInterface:
    """
    Given the evaluator name, load the evaluator
    """
    return EVALUATORS_DICT[evaluator_name](model, **kwargs)
