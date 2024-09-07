"""
The main training code
"""
import os

import hydra

from models.build_models import build_model
from trainers.build_trainers import build_trainer, ddp_setup
from trainers import base_trainer
from trainers.utils import create_folder_structure, init_print_override, restore_print_override
from models.utils import print_model_stats

import torch
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
from trainers.prepare import prepare_data

def ddp_main(rank, world_size, cfg):
    """
    Main function for distributed training
    """
    os.environ["GLOBAL_RANK"] = str(rank)

    original_print = init_print_override()

    try:
        print("Rank: ", rank, "World Size: ", world_size)
        ddp_setup(rank=rank, world_size=world_size)

        model = build_model(model_cfg=cfg["model"])
        model.to(cfg["general"]["device"])
        model.train()
        print(f"Rank{rank} Model built")
        print_model_stats(model)
        # load the relevant trainer
        trainer: base_trainer.BaseTrainer = build_trainer(
            cfg=cfg,
            model=model,
            gpu_id=rank
        )
        print(f"Rank{rank} Trainer built")
        # train the model
        trainer.train()
    
    finally:
        # clean up
        destroy_process_group()

        # restore the print function
        restore_print_override(original_print)

def basic_main(cfg):
    """
    Main function for single GPU training
    """
    model = build_model(model_cfg=cfg["model"])
    model.to(cfg["general"]["device"])
    model.train()
    print("Model built")
    # load the relevant trainer
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        gpu_id=None # disables DDP
    )

    # train the model
    trainer.train()


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    world_size = torch.cuda.device_count()
    
    if "full_configs" in cfg:
        cfg = cfg["full_configs"]
    cfg["general"]["paths"]["data_dir"] = hydra.utils.to_absolute_path(
        cfg["general"]["paths"]["data_dir"]
    ) # must be done before multiprocessing or else the path is wrong?
    cfg["general"]["paths"]["eval_dir"] = hydra.utils.to_absolute_path(
        cfg["general"]["paths"]["eval_dir"]
    )

    create_folder_structure(path_config=cfg["general"]["paths"])

    # process data 
    prepare_data(cfg)

    if world_size <= 1:
        # single GPU/CPU training
        basic_main(cfg)

    else:
        # multi-GPU training
        mp.spawn(
            ddp_main,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True,
        )

        # Additional cleanup to prevent leaked semaphores
        for process in mp.active_children():
            process.terminate()
            process.join()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
