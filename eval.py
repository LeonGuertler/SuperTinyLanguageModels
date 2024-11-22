"""
The main eval code
"""

import hydra
import torch

# from evals.load_evaluators import load_evaluator
from models.build_models import build_model
import evals 

@hydra.main(config_path="configs", config_name="test/test_stlm")
def main(cfg):
    """run the main eval loop"""

    # load checkpoint from the path if there
    if "model_ckpt" in cfg:
        # set the checkpoint path to absolute path
        cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])

        model, _ = build_model(checkpoint=torch.load(cfg["model_ckpt"]))
    # otherwise build the model from scratch (e.g. for external pretrained models)
    else:
        model, _ = build_model(model_cfg=cfg["model"])

    model.eval()

    # load the evaluator
    benchmark_names = cfg["testing"]["benchmarks"]
    benchmark_names = [str(benchmark_name) for benchmark_name in benchmark_names]

    results_list = []
    for benchmark_name in benchmark_names:
        # make benchmark
        benchmark_evaluator = evals.make(benchmark_name)

        # Eval
        results = benchmark_evaluator.evaluate(model=model)
        results_list.append(results)
        print(benchmark_name, results)
    print(results_list)
    # run the evaluator
    # results = evaluator.evaluate()
    # with open(cfg["output_path"], "w") as f:
    #     f.write(str(results))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
