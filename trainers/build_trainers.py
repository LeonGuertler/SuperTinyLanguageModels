"""
Builds the individual components of the trainer,
and the trainer itself.
"""

import os

import torch
from torch.distributed import init_process_group

from models.experimental.hugging_face import MockTrainer
from trainers import config, optimizers, schedulers
from trainers.base_trainer import BaseTrainer
from trainers.datasets import (
    BaseDataset,
    BytePoolingDataset,
    DatasetInterface,
    DualBytePooling,
)
from trainers.loss_fn import (
    cross_entropy_loss_fn,
    masked_cross_entropy_loss_fn,
    next_token_mlm_loss_fn,
)
from trainers.samplers import BaseSampler


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


def build_optimizer(model, optimizer_config: optimizers.OptimizerConfig):
    """
    Given the optimizer config, build the optimizer
    """
    match optimizer_config.name:
        case optimizers.OptimizerTypeNames.NANOGPT_ADAMW:
            optimizer_config: optimizers.NanoGPTAdamWConfig = optimizer_config
            return optimizers.configure_nanoGPT_optimizer(
                model=model,
                optimizer_cfg=optimizer_config,
            )
        case optimizers.OptimizerTypeNames.ADAMW:
            optimizer_config: optimizers.AdamWConfig = optimizer_config
            return torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_config.lr,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                weight_decay=optimizer_config.weight_decay,
            )


def build_lr_scheduler(scheduler_cfg: schedulers.LRSchedulerConfig):
    """
    Given the trainer config, build the LR scheduler.build_model
    """
    match scheduler_cfg.lr_scheduler_type:
        case schedulers.LRSchedulerNames.CONSTANT:
            return schedulers.LRScheduler(lr_scheduler_cfg=scheduler_cfg)
        case schedulers.LRSchedulerNames.COSINE:
            return schedulers.CosineLRScheduler(lr_scheduler_cfg=scheduler_cfg)


def build_dropout_scheduler(scheduler_cfg: schedulers.DropoutSchedulerConfig):
    """
    Given the trainer config, build the dropout scheduler.
    """
    match scheduler_cfg.dropout_type:
        case "constant":
            scheduler_cfg: schedulers.DropoutSchedulerConfig = scheduler_cfg
            return schedulers.DropoutScheduler(dropout_cfg=scheduler_cfg)
        case "linear":
            scheduler_cfg: schedulers.LinearDropoutSchedulerConfig = scheduler_cfg
            return schedulers.LinearDropoutScheduler(
                dropout_cfg=scheduler_cfg,
            )
        case "triangle":
            scheduler_cfg: schedulers.TriangleDropoutSchedulerConfig = scheduler_cfg
            return schedulers.TriangleDropoutScheduler(
                dropout_cfg=scheduler_cfg,
            )


DATASET_DICT: dict[str, DatasetInterface] = {
    "standard": BaseDataset,
    "byte_pooling": BytePoolingDataset,
    "dual_byte_pooling": DualBytePooling,
}


def build_dataset(cfg, split) -> BaseDataset:
    """
    Given the config, build the dataloader
    """
    return DATASET_DICT[cfg.trainer["dataloader"]["name"]](cfg=cfg, split=split)


DATASAMPLER_DICT = {"standard": BaseSampler}


def build_datasampler(dataset, sampling, batch_size) -> BaseSampler:
    """
    Given the dataset and the sampling method, build the dataloader
    """
    return DATASAMPLER_DICT[sampling](
        data_source=dataset,
        batch_size=batch_size,
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


def build_trainer(cfg: config.TrainConfig, model, gpu_id):
    """
    Given a config, this function builds a trainer
    and all relevant components of it.
    """

    # build optimizer
    optimizer = build_optimizer(model=model, optimizer_config=cfg.optimizer)

    # build LR scheduler
    lr_scheduler = build_lr_scheduler(scheduler_cfg=cfg.lr_scheduler)

    # build dropout scheduler
    dropout_scheduler = build_dropout_scheduler(scheduler_cfg=cfg.dropout_scheduler)

    # build dataloder
    train_dataset = build_dataset(cfg=cfg, split="train")
    val_dataset = build_dataset(cfg=cfg, split="val")

    # initialize datasamplers
    train_data_sampler = build_datasampler(
        dataset=train_dataset,
        sampling=cfg["trainer"]["datasampling"]["name"],
        batch_size=cfg["trainer"]["training"]["batch_size"]
        * cfg["trainer"]["training"]["gradient_accumulation_steps"],
    )
    val_data_sampler = build_datasampler(
        dataset=val_dataset,
        sampling=cfg["trainer"]["datasampling"]["name"],
        batch_size=cfg["trainer"]["training"]["batch_size"]
        * cfg["trainer"]["training"]["gradient_accumulation_steps"],
    )

    # wrap in dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_data_sampler,
        num_workers=1,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size,
        sampler=val_data_sampler,
        num_workers=1,
    )

    # build loss function
    loss_fn = build_loss_fn(loss_fn_name=cfg.loss_fn.loss_fn_type)

    # build the trainer
    print(cfg.training.trainer_type)
    trainer = TRAINER_DICT[cfg.training.trainer_type](
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dropout_scheduler=dropout_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        gpu_id=gpu_id,
    )

    return trainer
