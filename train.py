"""
The main training code
"""

import hydra

from models.build_model import build_model
from trainers.build_trainers import build_trainer
from trainers.utils import create_folder_structure


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    """Creates folder structure as necessary, and runs train"""
    # set data path to absolute path
    cfg["general"]["paths"]["data_path"] = hydra.utils.to_absolute_path(
        cfg["general"]["paths"]["data_path"]
    )
    # create necessary folder structure
    create_folder_structure(path_config=cfg["general"]["paths"])

    model = build_model(model_cfg=cfg["model"])
    model.to(cfg["general"]["device"])

    # load the relevant trainer
    trainer = build_trainer(
        cfg=cfg,
        model=model,
    )
    # preprocess the training data
    trainer.preprocess_data()

    # train the model
    trainer.train()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
