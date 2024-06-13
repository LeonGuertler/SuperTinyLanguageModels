"""Cascade Trainer, minor modification of behaviour of the _run_step func"""

from trainers.base_trainer import BaseTrainer
import torch
from models.experimental.cascade_ntp import cascade_shell

class CascadeTrainer(BaseTrainer):
    """
    Cascade Trainer, minor modification of behaviour of the _run_step func
    # TODO remove this trainer and just have the cascade loss function where its needed
    """

    def _run_step(self):
        """Run a single step of training"""
        assert isinstance(self.model, cascade_shell.CascadeShell)
        for _ in range(self.gradient_accumulation_steps):
            x, y = self.dataloader.get_batch("train")
            with self.ctx:
                output, aux_loss = self.model(x)
                loss = cascade_shell.compute_cascade_loss(output, y)
                if aux_loss is not None:
                    loss += aux_loss
                loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        grad_clip = self.cfg.trainer.optimizer.grad_clip
        if grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                grad_clip,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)
        return loss

    def estimate_performance(self, eval_iters=None):
        return super().estimate_performance(eval_iters)