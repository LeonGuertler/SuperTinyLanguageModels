"""Cascade Trainer, minor modification of behaviour of the _run_step func"""

from trainers.base_trainer import BaseTrainer
from models.experimental.cascade_ntp import cascade_shell

class CascadeTrainer(BaseTrainer):
    """
    Cascade Trainer, minor modification of behaviour of the _run_step func
    # TODO remove this trainer and just have the cascade loss function where its needed
    """

    def __init__(self,
        cfg,
        model,
        optimizer,
        dataloader,
        loss_fn,
        gpu_id, 
        lr_scheduler=None,
        dropout_scheduler=None,):
        super().__init__(cfg, model, optimizer, dataloader, loss_fn, gpu_id, lr_scheduler, dropout_scheduler)
        self.loss_fn = cascade_shell.compute_cascade_loss