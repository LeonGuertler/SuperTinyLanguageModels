"""Various Scheduler"""

import math
import torch.nn as nn

class LRScheduler:

    def __init__(self, lr):
        self.lr = lr

    def get_lr(self, iter_num):
        """Return Constant LR"""
        return self.lr


    def step(self, optimizer, iter_num):
        """Step the scheduler"""
        lr = self.get_lr(iter_num)
        self.apply_lr(optimizer, lr)
        return lr

class CosineLRScheduler(LRScheduler):
    """Basic Cosine LR scheduler with warmup and decay."""

    def __init__(self, warmup_iters, decay_iters, lr, min_lr):
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

    def apply_lr(self, optimizer, lr):
        """Apply the learning rate to the optimizer"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class DropoutScheduler:
    """Dropout Scheduler"""

    def __init__(self, dropout_p=0.1):
        self.dropout_p = dropout_p

    def get_dropout(self, iter):
        """Return Constant Dropout"""
        return self.dropout_p

    def set_dropout(self, model, dropout_p):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_p

    def step(self, model, iter):
        dropout_p = self.get_dropout(iter)
        self.set_dropout(model, dropout_p)
        return dropout_p

class LinearDropoutScheduler(DropoutScheduler):
    """Dropout Scheduler"""

    def __init__(self, start_iter, end_iter, start_dropout_p, end_dropout_p):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_dropout_p = start_dropout_p
        self.end_dropout_p = end_dropout_p

    def get_dropout(self, iter):
        """Return Constant Dropout"""
        if iter < self.start_iter:
            return self.start_dropout_p
        if iter >= self.end_iter:
            return self.end_dropout_p
        return self.start_dropout_p + (iter - self.start_iter) * (self.end_dropout_p - self.start_dropout_p) / (self.end_iter - self.start_iter)

