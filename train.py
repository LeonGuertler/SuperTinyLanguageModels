"""
The main training code
"""
import os

import hydra

from models.build_models import build_model
from trainers.build_trainers import build_trainer, ddp_setup
from trainers.utils import create_folder_structure, init_print_override, restore_print_override
from models.utils import print_model_stats
from trainers.build_teachermodel import init_teachermodel

from omegaconf import OmegaConf

import torch
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp

def ddp_main(rank, world_size, cfg):
    """
    Main function for distributed training
    """
    os.environ["GLOBAL_RANK"] = str(rank)

    original_print = init_print_override()

    try:
        print("Rank: ", rank, "World Size: ", world_size)
        ddp_setup(rank=rank, world_size=world_size)
        
        ## Locate the model checkpoint
        if cfg.get("model_ckpt", None):
            cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])
            checkpoint = torch.load(cfg["model_ckpt"])
            ## add the previous iteration number to the config as a new key
            dict_cfg = OmegaConf.to_container(cfg, resolve=True)
            dict_cfg['prev_iter'] = checkpoint['config']['trainer']['training']['max_iters']
            cfg = OmegaConf.create(dict_cfg)

            model = build_model(checkpoint=checkpoint)
        else: 
            model = build_model(model_cfg=cfg["model"])

        model.to(cfg["general"]["device"])
        model.train()
        print(f"Rank{rank} Model built")
        print_model_stats(model)

        # load the teacher model, if any
        if cfg.get("teachermodel", None):
            teacher_model, projection = init_teachermodel(cfg)
        else:
            teacher_model, projection = None, None

        # load the relevant trainer
        trainer = build_trainer(
            cfg=cfg,
            model=model,
            gpu_id=rank,
            projection=projection,
            teacher_model=teacher_model,
        )

        print(f"Rank{rank} Trainer built")
        # preprocess the training data
        trainer.preprocess_data()
        print(f"Rank{rank} Data preprocessed")
        # train the model
        trainer.train()
    
    finally:
        # clean up
        destroy_process_group()

        # restore the print function
        restore_print_override(original_print)


# @hydra.main(config_path="configs/full_configs", config_name="knowledge_distillation")
@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    world_size = torch.cuda.device_count()
    
    if "full_configs" in cfg:
        cfg = cfg["full_configs"]
    cfg["general"]["paths"]["data_dir"] = hydra.utils.to_absolute_path(
        cfg["general"]["paths"]["data_dir"]
    ) # must be done before multiprocessing or else the path is wrong?

    create_folder_structure(path_config=cfg["general"]["paths"])
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
