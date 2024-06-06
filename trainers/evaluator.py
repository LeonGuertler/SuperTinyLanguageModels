"""Code for running samples from the evaluation benchmarks"""

from evals.load_evaluators import load_evaluator


def train_eval(eval_cfg, model):
    """Train the model"""
    evaluator = load_evaluator(eval_cfg["evaluator"], model)
    results = evaluator.evaluate(
        **eval_cfg
    )
    return results
