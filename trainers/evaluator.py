"""Code for running samples from the evaluation benchmarks"""

from evals.load_evaluators import load_evaluator

def train_eval(eval_cfg, model):
    """Train the model"""
    evaluator_name = eval_cfg["evaluator"]
    kwargs = {
        key: value for key, value in eval_cfg.items() if key != "evaluator"
    }
    evaluator = load_evaluator(evaluator_name, model, **kwargs)
    results = evaluator.evaluate()
    return results


def train_eval_text_modeling(model, eval_dir):
    """ Test the model """
    evaluator = load_evaluator("text_modeling", model, eval_dir=eval_dir)
    results = evaluator.evaluate()
    return results