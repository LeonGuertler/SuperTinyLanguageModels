"""Code for running samples from the evaluation benchmarks"""

from evals.load_evaluators import load_evaluator

def train_eval(eval_cfg, model):
    """Train the model"""
    evaluator = load_evaluator(eval_cfg["evaluator"], model)
    results = {}
    for benchmark in eval_cfg["benchmarks"]:
        results[benchmark] = (
            evaluator.evaluate_benchmark(benchmark, num_samples=eval_cfg["num_samples"])
        )
    return results