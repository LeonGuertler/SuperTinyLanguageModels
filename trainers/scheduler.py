"""Various Scheduler"""

import math

import torch.nn as nn


class LRScheduler:
    """Constant LR scheduler"""

    def __init__(self, lr):
        self.lr = lr

    def get_lr(self, _):
        """Return Constant LR"""
        return self.lr

    def step(self, optimizer, iter_num):
        """Step the scheduler"""
        lr = self.get_lr(iter_num)
        self.apply_lr(optimizer, lr)
        return lr

    def apply_lr(self, optimizer, lr):
        """Apply the learning rate to the optimizer"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class CosineLRScheduler(LRScheduler):
    """Basic Cosine LR scheduler with warmup and decay."""

    def __init__(self, warmup_iters, decay_iters, lr, min_lr):
        """Initialize the scheduler"""
        super().__init__(lr)
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.lr = lr
        self.min_lr = min_lr

    def get_lr(self, iter_num):
        """Get the learning rate for the iteration number"""
        if iter_num < self.warmup_iters:
            return self.lr * iter_num / self.warmup_iters
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + math.cos((iter_num - self.warmup_iters) / self.decay_iters * math.pi)
        )
