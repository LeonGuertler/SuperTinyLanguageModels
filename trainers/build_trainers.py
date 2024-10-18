"""
Builds the individual components of the trainer,
and the trainer itself.
"""

import os

import torch
from torch.distributed import init_process_group

from models.experimental.hugging_face import MockTrainer
from trainers.base_trainer import BaseTrainer
from trainers.datasets import build_dataset
from trainers.loss_fn import (
    cross_entropy_loss_fn,
    next_token_mlm_loss_fn,
)
from trainers.optimizers import build_optimizer
from trainers.schedulers import build_scheduler

from trainers.data_utils import (
    collate_fn,
    identify_collate_fn
)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # Get the master address and port from SLURM environment variables
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    # Set the environment variables for PyTorch distributed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)





# SCHEDULER_DICT = {
#     "cosine": lambda trainer_cfg: CosineLRScheduler(
#         warmup_iters=trainer_cfg["lr_scheduler"]["warmup_iters"],
#         decay_iters=trainer_cfg["lr_scheduler"].get(
#             "lr_decay_iters", 
#             trainer_cfg["max_iters"]
#         ),
#         lr=trainer_cfg["optimizer_params"]["lr"],
#         min_lr=trainer_cfg["lr_scheduler"]["min_lr"],
#     ),
#     "constant": lambda trainer_cfg: LRScheduler(
#         lr=trainer_cfg["optimizer_params"]["lr"],
#     ),
# }


# def build_lr_scheduler(trainer_cfg):
#     """
#     Given the trainer config, build the LR scheduler.build_model
#     """
#     return SCHEDULER_DICT[trainer_cfg["lr_scheduler"]["name"]](trainer_cfg=trainer_cfg)




# DATASET_DICT: dict[str, DatasetInterface] = {
#     "standard": BaseDatasetRandom,
#     "byte_pooling": BytePoolingDataset,
#     "dual_byte_pooling": DualBytePooling,
# }


# def build_dataset(cfg, split):
#     """
#     Given the config, build the dataloader
#     """
#     return DATASET_DICT[cfg.trainer["dataloader"]["name"]](cfg=cfg, split=split)



LOSS_FN_DICT = {
    "cross_entropy": cross_entropy_loss_fn,
    "next_token_mlm": next_token_mlm_loss_fn,
}


def build_loss_fn(loss_fn_name):
    """
    Given the loss function name, build the loss function
    """
    return LOSS_FN_DICT[loss_fn_name]


TRAINER_DICT = {
    "base_trainer": BaseTrainer,
    "mock_trainer": MockTrainer,
}


def build_trainer(cfg, model, gpu_id, loaded_train_config):
    """
    Given a config, this function builds a trainer
    and all relevant components of it.
    """

    # build optimizer
    optimizer = build_optimizer(
        optimizer_name=cfg.trainer["optimizer_name"],
        model=model,
        optimizer_params=cfg.trainer["optimizer_params"]
    )

    # build LR scheduler
    lr_scheduler = build_scheduler(
        scheduler_name=cfg.trainer["lr_scheduler"]["name"],
        optimizer=optimizer,
        scheduler_params=cfg.trainer["lr_scheduler"]["params"],    
    )

    # build dataloder
    train_dataset = build_dataset(cfg=cfg, split="train")
    val_dataset = build_dataset(cfg=cfg, split="val")

    # wrap in dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn if cfg.trainer.get("use_collate_fn", False) else identify_collate_fn,

    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn if cfg.trainer.get("use_collate_fn", False) else identify_collate_fn,
    )

    # build loss function
    loss_fn = build_loss_fn(loss_fn_name=cfg.trainer["loss_fn"]["name"])

    # build the trainer
    print(cfg.trainer["trainer_type"])
    trainer = TRAINER_DICT[cfg.trainer["trainer_type"]](
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        gpu_id=gpu_id,
        loaded_train_config=loaded_train_config,
    )

    return trainer
