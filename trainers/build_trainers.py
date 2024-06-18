"""
Builds the individual components of the trainer,
and the trainer itself.
"""

from models.experimental.hugging_face import MockTrainer
from trainers.base_trainer import BaseTrainer
from trainers.dataloader import (
    BaseDataloader,
    BytePoolingDataloader,
    NextTokenMLMDataloader,
    ConversationalDataloader,
)
from trainers.loss_fn import (
    cross_entropy_loss_fn,
    next_token_mlm_loss_fn,
    masked_cross_entropy_loss_fn
)
from trainers.optimizer import configure_nanoGPT_optimizer
from trainers.scheduler import (
    CosineLRScheduler,
    DropoutScheduler,
    LinearDropoutScheduler,
    LRScheduler,
    TriangleDropoutScheduler
)

from trainers.prepare import prepare_data

import torch
from torch.distributed import init_process_group
import os

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
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
            max_iterations=trainer_cfg["training"]["max_iters"],
            gradient_accumulated_steps=trainer_cfg["training"]["gradient_accumulation_steps"],
            cycle_factor=trainer_cfg["dropout_scheduler"]["cycle_factor"],
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


def build_dataloader(cfg, split):
    """
    Given the config, build the dataloader
    """
    return DATALOADER_DICT[cfg.trainer["dataloader"]["name"]](
        cfg=cfg,
        split=split
    )


DATADAMPLER_DICT = {
    "standard": torch.utils.data.DataLoader
}

def build_datasampler(dataset, sampling):
    """
    Given the dataset and the sampling method, build the dataloader
    """
    return DATADAMPLER_DICT[sampling](dataset)

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

    # prepare data
    prepare_data(cfg, model.embedding_model)

    # build dataloder
    train_dataset = build_dataloader(cfg=cfg, split="train")
    val_dataset = build_dataloader(cfg=cfg, split="val")

    # downsample the val_dataset to eval iters
    val_dataset = val_dataset.downsample(cfg["trainer"]["training"]["eval_iters"])

    # wrap both in dataloaders
    train_dataloader = build_datasampler(
        dataset=train_dataset,
        sampling=cfg["trainer"]["datasampling"]["name"]
    )
    val_dataloader = build_datasampler(
        dataset=val_dataset,
        sampling=cfg["trainer"]["datasampling"]["name"]
    )

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
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        gpu_id=gpu_id
    )

    return trainer
