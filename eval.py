"""
The main eval code
"""
import os
import hydra
import torch
from evals import build_benchmark
from evals import model_wrapper
from models import build_models
from models import generator
from trainers.utils import (
    create_folder_structure,
)


@hydra.main(config_path="configs", config_name="evals")
def main(cfg):
    """Creates folder structure as necessary, and runs train"""
    # print(cfg["train"])
    # set data path to absolute path
    cfg["train"]["general"]["paths"]["data_path"] = hydra.utils.to_absolute_path(
        cfg["train"]["general"]["paths"]["data_path"]
    )
    cfg["checkpoint_path"] = hydra.utils.to_absolute_path(cfg["checkpoint_path"])
    # load checkpoint from the path
    model_checkpoint = torch.load(cfg["checkpoint_path"])
    model_dict = cfg["train"]["model"]
    model = build_models.build_model(
        cfg=model_dict,
        model_checkpoint=model_checkpoint
    )
    model = generator.build_generator(model=model, generate_cfg=cfg["generate_config"])
    model = model_wrapper.ModelWrapper(model=model)

    # create necessary folder structure
    create_folder_structure(path_config=cfg["train"]["general"]["paths"])

    for benchmark_name in cfg["benchmarks"]:
        # load the relevant class:
        print(cfg["train"]["general"])
        benchmark = build_benchmark.build_benchmark(
            benchmark_name=benchmark_name, model=model
        )
        print(benchmark.execute())


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
