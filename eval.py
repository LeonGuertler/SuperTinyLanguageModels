"""
The main eval code
"""

import hydra
import torch

from evals.load_evaluators import load_evaluator
from models.build_models import build_model


@hydra.main(config_path="configs/eval", config_name="baseline")
def main(cfg):
    """run the main eval loop"""

    # set the checkpoint path to absolute path
    cfg["ckpt_path"] = hydra.utils.to_absolute_path(cfg["ckpt_path"])

    # load checkpoint from the path
    model = build_model(checkpoint=torch.load(cfg["ckpt_path"]))

    # load the evaluator
    evaluator = load_evaluator(
        evaluator_name=cfg["evaluator_name"], cfg=cfg, model=model
    )

    # run the evaluator
    evaluator.evaluate(benchmark_names=cfg["benchmarks"])


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
    # outputs/2024-04-11/11-34-28/checkpoints/ckpt_9999.pt
