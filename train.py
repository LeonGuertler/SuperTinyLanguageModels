import os
import hydra
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

from models.build_models import build_model
from trainers.build_trainers import build_trainer, ddp_setup
from trainers.utils import create_folder_structure, init_print_override, restore_print_override
from trainers.prepare import prepare_data

def single_gpu_training(cfg):
    """
    Function to handle training on a single GPU
    """
    original_print = init_print_override()

    try:
        print("Training on a single GPU")
        model = build_model(model_cfg=cfg["model"])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        trainer = build_trainer(cfg=cfg, model=model, gpu_id=0)
        trainer.train()

    finally:
        restore_print_override(original_print)

def ddp_main(rank, world_size, cfg):
    """
    Main function for distributed training
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["GLOBAL_RANK"] = str(rank)

    original_print = init_print_override()

    try:
        print("Rank: ", rank, "World Size: ", world_size)
        ddp_setup(rank=rank, world_size=world_size)

        model = build_model(model_cfg=cfg["model"])
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        model.train()
        print(f"Rank {rank} Model built")

        trainer = build_trainer(cfg=cfg, model=model, gpu_id=rank)
        print(f"Rank {rank} Trainer built")

        trainer.train()
    
    finally:
        destroy_process_group()
        restore_print_override(original_print)

@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(cfg):
    world_size = torch.cuda.device_count()

    if "full_configs" in cfg:
        cfg = cfg["full_configs"]

    cfg["general"]["paths"]["data_dir"] = hydra.utils.to_absolute_path(cfg["general"]["paths"]["data_dir"])
    create_folder_structure(path_config=cfg["general"]["paths"])

    print('Preparing training data')
    prepare_data(cfg=cfg)

    if world_size > 1:
        # Use distributed training
        mp.spawn(ddp_main, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        # Fallback to single GPU training
        single_gpu_training(cfg)

    # Cleanup potentially leaked processes
    for process in mp.active_children():
        process.terminate()
        process.join()

if __name__ == "__main__":
    main()
