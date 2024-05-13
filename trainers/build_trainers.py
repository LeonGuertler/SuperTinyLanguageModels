"""
Builds the individual components of the trainer,
and the trainer itself.
"""

from models.build_models import build_model
from trainers.base_profiler import BaseProfiler

# from trainers.standard_trainer import BaseTrainer
from trainers.base_trainer import BaseTrainer
from trainers.byte_trainer import ByteTrainer
from trainers.dataloader import (
    BytePoolingDataloader,
    Seq2SeqDataloader,
    StandardDataloader,
)
from trainers.loss_fn import cross_entropy_loss_fn
from trainers.optimizer import configure_nanoGPT_optimizer
from trainers.scheduler import (
    CosineLRScheduler,
    DropoutScheduler,
    LinearDropoutScheduler,
    LRScheduler,
)

OPTIMIZER_DICT = {
    "nanoGPTadamW": lambda model, cfg: configure_nanoGPT_optimizer(
        model=model,
        weight_decay=cfg["weight_decay"],
        learning_rate=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )
}


def build_optimizer(model, optimizer_config):
    """
    Given the optimizer config, build the optimizer
    """
    return OPTIMIZER_DICT[optimizer_config["name"]](model=model, cfg=optimizer_config)


SCHEDULER_DICT = {
    "cosine": lambda cfg: CosineLRScheduler(
        warmup_iters=cfg["training"]["warmup_iters"],
        decay_iters=cfg["training"]["lr_decay_iters"],
        lr=cfg["optimizer"]["lr"],
        min_lr=cfg["optimizer"]["min_lr"],
    ),
    "constant": lambda cfg: LRScheduler(
        lr=cfg["optimizer"]["lr"],
    ),
}


def build_lr_scheduler(trainer_cfg):
    """
    Given the trainer config, build the LR scheduler.build_model
    """
    return SCHEDULER_DICT[trainer_cfg["lr_scheduler"]["name"]](cfg=trainer_cfg)


def build_dropout_scheduler(trainer_cfg):
    """
    Given the trainer config, build the dropout scheduler.
    """
    if trainer_cfg["dropout_scheduler"]["name"] == "constant":
        return DropoutScheduler(trainer_cfg["dropout_scheduler"]["dropout"])
    if trainer_cfg["dropout_scheduler"]["name"] == "linear":
        return LinearDropoutScheduler(
            start_dropout_p=trainer_cfg["dropout_scheduler"]["start_dropout_p"],
            end_dropout_p=trainer_cfg["dropout_scheduler"]["end_dropout_p"],
            start_iter=trainer_cfg["dropout_scheduler"]["start_iter"],
            end_iter=trainer_cfg["dropout_scheduler"]["end_iter"],
        )
    raise NotImplementedError(
        f"dropout scheduler {trainer_cfg['dropout_scheduler']['name']} not implemented."
    )


DATALODER_DICT = {
    "standard": StandardDataloader,
    "byte_pooling_dataloader": BytePoolingDataloader,
    "seq2seq": Seq2SeqDataloader,
}


def build_dataloader(cfg):
    """
    Given the config, build the dataloader
    """
    return DATALODER_DICT[cfg["trainer"]["dataloader"]["name"]](
        cfg=cfg,
        data_dir=cfg["general"]["paths"]["data_path"],
    )


LOSS_FN_DICT = {"cross_entropy": cross_entropy_loss_fn}


def build_loss_fn(loss_fn_name):
    """
    Given the loss function name, build the loss function
    """
    return LOSS_FN_DICT[loss_fn_name]


TRAINER_DICT = {
    "base_trainer": BaseTrainer,
    "byte_trainer": ByteTrainer,
    "base_profiler": BaseProfiler,
}


def build_trainer(cfg):
    """
    Given a config, this function builds a trainer
    and all relevant components of it.
    """

    # build model
    model = build_model( 
        model_cfg=cfg["model"],
    )

    # push model to device
    model.to(cfg["general"]["device"])

    # build optimizer
    optimizer = build_optimizer(
        model=model, optimizer_config=cfg["trainer"]["optimizer"]
    )

    # build LR scheduler
    lr_scheduler = build_lr_scheduler(trainer_cfg=cfg["trainer"])

    # build dropout scheduler
    dropout_scheduler = build_dropout_scheduler(trainer_cfg=cfg["trainer"])

    # build dataloder
    dataloader = build_dataloader(cfg=cfg)

    # build loss function
    loss_fn = build_loss_fn(loss_fn_name=cfg["trainer"]["loss_fn"]["name"])

    # build the trainer
    print(cfg["trainer"]["training"]["trainer"])
    trainer = TRAINER_DICT[cfg["trainer"]["training"]["trainer"]](
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dropout_scheduler=dropout_scheduler,
        dataloader=dataloader,
        loss_fn=loss_fn,
    )

    return trainer
