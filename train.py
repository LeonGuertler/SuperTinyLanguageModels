"""
The main training code
"""

import hydra

from models.build_models import build_model
from trainers.build_trainers import build_trainer, ddp_setup
from trainers.utils import create_folder_structure

import torch
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp

def ddp_main(rank, world_size, cfg):
    """
    Main function for distributed training
    """
    ddp_setup(rank=rank, world_size=world_size)

    if "full_configs" in cfg:
        cfg = cfg["full_configs"]
    cfg["general"]["paths"]["data_dir"] = hydra.utils.to_absolute_path(
        cfg["general"]["paths"]["data_dir"]
    )
    # create necessary folder structure
    create_folder_structure(path_config=cfg["general"]["paths"])

    model = build_model(model_cfg=cfg["model"])
    model.to(cfg["general"]["device"])
    model.train()
    
    # load the relevant trainer
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        gpu_id=rank
    )
    # preprocess the training data
    trainer.preprocess_data()

    # train the model
    trainer.train()

    # clean up
    destroy_process_group()


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    world_size = torch.cuda.device_count()
    mp.spawn(
        ddp_main,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
