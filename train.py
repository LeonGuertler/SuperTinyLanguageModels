"""
The main training code
"""

import hydra

from trainers.build_trainers import build_trainer
from trainers.utils import create_folder_structure


@hydra.main(config_path="configs/train", config_name="baseline")
def main(cfg):
    """Creates folder structure as necessary, and runs train"""
    # create necessary folder structure
    create_folder_structure(path_config=cfg["general"]["paths"])

    # load the relevant trainer
    trainer = build_trainer(
        cfg=cfg,
    )
    # preprocess the training data
    trainer.preprocess_data()

    # train the model
    trainer.train()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
