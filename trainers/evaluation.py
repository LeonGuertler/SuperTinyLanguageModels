"""Code for running samples from the evaluation benchmarks"""

import pydantic

from evals.load_evaluators import load_evaluator


class EvaluationConfig(pydantic.BaseModel):
    """Configuration for Evaluation during training"""

    eval_interval: int = 2000
    eval_iters: int = 500
    evaluators: list[dict]


class EvaluatorConfig(pydantic.BaseModel):
    """Configuration for Evaluation during training"""

    evaluator: str


class MCQEvaluatorConfig(EvaluatorConfig):
    """Configuration for Multiple Choice Question Evaluation"""

    evaluator: str = "mcq"
    benchmarks: list[str] = ["winograd", "hellaswag", "arc", "mmlu", "blimp"]
    num_samples: int = 1000


class PROGEvaluatorConfig(EvaluatorConfig):
    """Configuration for PROG Evaluation"""

    evaluator: str = "prog"


def get_evaluator_config(evaluator_dict):
    """Get the evaluator config"""
    evaluator_name = evaluator_dict["evaluator"]
    if evaluator_name == "mcq":
        return MCQEvaluatorConfig(**evaluator_dict)
    elif evaluator_name == "prog":
        return PROGEvaluatorConfig(**evaluator_dict)
    else:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")


def train_eval(eval_cfg: EvaluatorConfig, model):
    """Train the model"""
    evaluator_name = eval_cfg.evaluator

    if evaluator_name == "mcq":
        eval_cfg: MCQEvaluatorConfig = eval_cfg
        mcq_evaluator = load_evaluator(
            evaluator_name,
            model,
            benchmarks=eval_cfg.benchmarks,
            num_samples=eval_cfg.num_samples,
        )
        results = mcq_evaluator.evaluate()
        return results
    elif evaluator_name == "prog":
        eval_cfg: PROGEvaluatorConfig = eval_cfg
        prog_evaluator = load_evaluator(evaluator_name, model)
        results = prog_evaluator.evaluate()
        return results
    else:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")
