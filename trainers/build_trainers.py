"""
Builds the individual components of the trainer, 
and the trainer itself.
"""

from models.build_models import build_model
from trainers.optimizer import (
    configure_nanoGPT_optimizer
)
from trainers.scheduler import (
    CosineScheduler
)

# from trainers.standard_trainer import BaseTrainer
from trainers.base_trainer import BaseTrainer

from trainers.dataloader import (
    StandardDataloader
)

from trainers.loss_fn import (
    cross_entropy_loss_fn
)




OPTIMIZER_DICT = {
    "nanoGPTadamW": lambda model, cfg: configure_nanoGPT_optimizer(
        model=model,
        weight_decay=cfg["weight_decay"],
        learning_rate=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"])
    )
}
def build_optimizer(model, optimizer_config):
    """
    Given the optimizer config, build the optimizer
    """
    print(optimizer_config["name"])
    return OPTIMIZER_DICT[optimizer_config["name"]](
        model=model,
        cfg=optimizer_config
    )


SCHEDULER_DICT = {
    "cosine": lambda cfg: CosineScheduler(
        warmup_iters=cfg["training"]["warmup_iters"],
        decay_iters=cfg["training"]["lr_decay_iters"],
        lr=cfg["optimizer"]["lr"],
        min_lr=cfg["optimizer"]["min_lr"]
    )
}
def build_scheduler(trainer_cfg):
    """
    Given the trainer config, build the LR scheduler.build_model
    """
    return SCHEDULER_DICT[trainer_cfg["scheduler"]["name"]](
        cfg=trainer_cfg
    )

DATALODER_DICT = {
    "standard": StandardDataloader
}
def build_dataloader(cfg):
    """
    Given the config, build the dataloader
    """
    return DATALODER_DICT[cfg["trainer"]["dataloader"]["name"]](
        cfg=cfg,
        data_dir = cfg["general"]["paths"]["data_path"],
    )

LOSS_FN_DICT = {
    "cross_entropy": cross_entropy_loss_fn
}
def build_loss_fn(loss_fn_name):
    """
    Given the loss function name, build the loss function
    """
    return LOSS_FN_DICT[loss_fn_name]


def build_trainer(cfg):
    """
    Given a config, this function builds a trainer 
    and all relevant components of it.
    """

    # build model
    model_dict = cfg["model"]
    model = build_model(
        cfg=model_dict,
    )

    # push model to device
    model.to(cfg["general"]["device"])

    # build optimizer
    optimizer = build_optimizer(
        model=model,
        optimizer_config=cfg["trainer"]["optimizer"]
    )

    # build LR scheduler
    scheduler = build_scheduler(
        trainer_cfg=cfg["trainer"]
    )

    # build dataloder
    dataloader = build_dataloader(
        cfg=cfg
    )

    # build loss function
    loss_fn = build_loss_fn(
        loss_fn_name=cfg["trainer"]["loss_fn"]["name"]
    )


    # build the trainer
    trainer = BaseTrainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        loss_fn=loss_fn,
    )


    return trainer
