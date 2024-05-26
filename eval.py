"""
The main eval code
"""

import hydra
import torch

from evals.load_evaluators import load_evaluator
from models.build_models import build_model


@hydra.main(config_path="configs", config_name="test")
def main(cfg):
    """run the main eval loop"""

    # set the checkpoint path to absolute path
    cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])

    # load checkpoint from the path
    model = build_model(checkpoint=torch.load(cfg["model_ckpt"]))

    # load the evaluator
    evaluator = load_evaluator(
        evaluator_name=cfg["testing"]["evaluator_name"], model=model
    )

    # run the evaluator
    benchmark_names = cfg["testing"]["benchmarks"]
    benchmark_names = [str(benchmark_name) for benchmark_name in benchmark_names]
    results = evaluator.evaluate(benchmark_names=benchmark_names)
    with open(cfg["output_path"], "w") as f:
        f.write(str(results))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
