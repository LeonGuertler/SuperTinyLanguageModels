"""
Builds the individual components of the trainer,
and the trainer itself.
"""

import os

import torch
from torch.distributed import init_process_group

from models.experimental.hugging_face import MockTrainer
from trainers.base_trainer import BaseTrainer
from trainers.dataloader import (
    BaseDataloader,
    BytePoolingDataloader,
    ConversationalDataloader,
    NextTokenMLMDataloader,
)
from trainers.loss_fn import (
    cross_entropy_loss_fn,
    masked_cross_entropy_loss_fn,
    next_token_mlm_loss_fn,
)
from trainers.optimizer import configure_nanoGPT_optimizer
from trainers.scheduler import (
    CosineLRScheduler,
    DropoutScheduler,
    LinearDropoutScheduler,
    LRScheduler,
    TriangleDropoutScheduler,
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


OPTIMIZER_DICT = {
    "nanoGPTadamW": lambda model, trainer_cfg: configure_nanoGPT_optimizer(
        model=model,
        weight_decay=trainer_cfg["weight_decay"],
        learning_rate=trainer_cfg["lr"],
        betas=(trainer_cfg["beta1"], trainer_cfg["beta2"]),
    )
}


def build_optimizer(model, optimizer_config):
    """
    Given the optimizer config, build the optimizer
    """
    return OPTIMIZER_DICT[optimizer_config["name"]](
        model=model, trainer_cfg=optimizer_config
    )


SCHEDULER_DICT = {
    "cosine": lambda trainer_cfg: CosineLRScheduler(
        warmup_iters=trainer_cfg["training"]["warmup_iters"],
        decay_iters=trainer_cfg["training"]["lr_decay_iters"],
        lr=trainer_cfg["optimizer"]["lr"],
        min_lr=trainer_cfg["optimizer"]["min_lr"],
    ),
    "constant": lambda trainer_cfg: LRScheduler(
        lr=trainer_cfg["optimizer"]["lr"],
    ),
}


def build_lr_scheduler(trainer_cfg):
    """
    Given the trainer config, build the LR scheduler.build_model
    """
    return SCHEDULER_DICT[trainer_cfg["lr_scheduler"]["name"]](trainer_cfg=trainer_cfg)


def build_dropout_scheduler(trainer_cfg):
    """
    Given the trainer config, build the dropout scheduler.
    """
    if trainer_cfg["dropout_scheduler"]["dropout_type"] == "constant":
        return DropoutScheduler(trainer_cfg["dropout_scheduler"]["dropout"])
    if trainer_cfg["dropout_scheduler"]["dropout_type"] == "linear":
        return LinearDropoutScheduler(
            start_dropout_p=trainer_cfg["dropout_scheduler"]["start_dropout_p"],
            end_dropout_p=trainer_cfg["dropout_scheduler"]["end_dropout_p"],
            start_iter=trainer_cfg["dropout_scheduler"]["start_iter"],
            end_iter=trainer_cfg["dropout_scheduler"]["end_iter"],
        )
    if trainer_cfg["dropout_scheduler"]["dropout_type"] == "triangle":
        return TriangleDropoutScheduler(
            dropout_trough=trainer_cfg["dropout_scheduler"]["dropout_trough"],
            dropout_peak=trainer_cfg["dropout_scheduler"]["dropout_peak"],
            num_iterations=trainer_cfg["dropout_scheduler"]["num_iterations"],
            num_cycles=trainer_cfg["dropout_scheduler"]["num_cycles"],
        )
    raise NotImplementedError(
        f"dropout scheduler {trainer_cfg['dropout_scheduler']['dropout_type']} not implemented."
    )


DATALOADER_DICT: dict[str, BaseDataloader] = {
    "standard": BaseDataloader,
    "byte_pooling": BytePoolingDataloader,
    "next_token_mlm": NextTokenMLMDataloader,
    "conversational": ConversationalDataloader,
}


def build_dataloader(cfg, embedder):
    """
    Given the config, build the dataloader
    """
    return DATALOADER_DICT[cfg.trainer["dataloader"]["name"]](
        cfg=cfg,
        embedder=embedder,
    )


LOSS_FN_DICT = {
    "cross_entropy": cross_entropy_loss_fn,
    "next_token_mlm": next_token_mlm_loss_fn,
    "masked_cross_entropy": masked_cross_entropy_loss_fn,
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


def build_trainer(cfg, model, gpu_id):
    """
    Given a config, this function builds a trainer
    and all relevant components of it.
    """

    # build optimizer
    optimizer = build_optimizer(model=model, optimizer_config=cfg.trainer["optimizer"])

    # build LR scheduler
    lr_scheduler = build_lr_scheduler(trainer_cfg=cfg.trainer)

    # build dropout scheduler
    dropout_scheduler = build_dropout_scheduler(trainer_cfg=cfg.trainer)

    # build dataloder
    dataloader = build_dataloader(cfg=cfg, embedder=model.embedding_model)
    dataloader.prepare_data()

    # build loss function
    loss_fn = build_loss_fn(loss_fn_name=cfg.trainer["loss_fn"]["name"])

    # build the trainer
    print(cfg.trainer["training"]["trainer_type"])
    trainer = TRAINER_DICT[cfg.trainer["training"]["trainer_type"]](
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dropout_scheduler=dropout_scheduler,
        dataloader=dataloader,
        loss_fn=loss_fn,
        gpu_id=gpu_id,
    )

    return trainer
