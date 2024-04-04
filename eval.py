"""
The main eval code
"""

import hydra
from SuperTinyLanguageModels.eval import build_benchmark
from SuperTinyLanguageModels.eval import model_wrapper
from SuperTinyLanguageModels.models import build_models
from SuperTinyLanguageModels.models import generator
from trainers.utils import (
    create_folder_structure,
)


@hydra.main(config_path="configs", config_name="eval")
def main(cfg):
    """Creates folder structure as necessary, and runs train"""

    # set data path to absolute path
    cfg["train"]["general"]["paths"]["data_path"] = hydra.utils.to_absolute_path(
        cfg["train"]["general"]["paths"]["data_path"]
    )

    # create necessary folder structure
    create_folder_structure(path_config=cfg["train"]["general"]["paths"])

    for benchmark_name in cfg["eval"]["benchmarks"]:
        # load the relevant class:
        model_dict = cfg["model"]
        model = build_models.build_model(
            cfg=model_dict,
        )
        model = generator.build_generator(model=model, generate_cfg=cfg["generate"])
        model = model_wrapper.ModelWrapper(model=model)
        benchmark = build_benchmark.build_benchmark(
            benchmark_name=benchmark_name, model=model
        )
        benchmark.execute()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
