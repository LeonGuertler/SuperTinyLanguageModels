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


class DropoutScheduler:
    """Constant Dropout Scheduler"""

    def __init__(self, dropout_p=0.1):
        self.dropout_p = dropout_p

    def get_dropout(self, _):
        """Return Constant Dropout"""
        return self.dropout_p

    def set_dropout(self, model, dropout_p):
        """Set the dropout probability for the model"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_p

    def step(self, model, iter_num):
        """Step the scheduler"""
        dropout_p = self.get_dropout(iter_num)
        self.set_dropout(model, dropout_p)
        return dropout_p


class LinearDropoutScheduler(DropoutScheduler):
    """Dropout Scheduler"""

    def __init__(self, start_iter, end_iter, start_dropout_p, end_dropout_p):
        """Initialize the dropout schedule"""
        super().__init__(start_dropout_p)
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_dropout_p = start_dropout_p
        self.end_dropout_p = end_dropout_p

    def get_dropout(self, iter_num):
        """Return Constant Dropout"""
        if iter_num < self.start_iter:
            return self.start_dropout_p
        if iter_num >= self.end_iter:
            return self.end_dropout_p
        return self.start_dropout_p + (iter_num - self.start_iter) * (
            self.end_dropout_p - self.start_dropout_p
        ) / (self.end_iter - self.start_iter)


class TriangleDropoutScheduler(DropoutScheduler):
    """Triangle Dropout Scheduler. Ref: https://arxiv.org/pdf/1506.01186"""

    def __init__(
        self,
        dropout_trough,
        dropout_peak,
        num_iterations,
        num_cycles=4,
    ):
        """Initialize the dropout schedule
        Args:
            dropout_trough: The minimum dropout probability
            dropout_peak: The maximum dropout probability
            num_iterations: The total number of iterations
            num_cycles: The number of cycles"""
        super().__init__(dropout_trough)
        self.dropout_trough = dropout_trough
        self.dropout_peak = dropout_peak
        self.total_iterations = num_iterations
        self.cycle_length = self.total_iterations // num_cycles

    def get_dropout(self, iter_num):
        cycle_position = iter_num % self.cycle_length
        half_cycle = self.cycle_length / 2
        if cycle_position < half_cycle:
            return self.dropout_trough + (self.dropout_peak - self.dropout_trough) * (
                cycle_position / half_cycle
            )
        return self.dropout_peak - (self.dropout_peak - self.dropout_trough) * (
            (cycle_position - half_cycle) / half_cycle
        )

    def step(self, model, iter_num):
        dropout_p = self.get_dropout(iter_num)
        self.set_dropout(model, dropout_p)
        return dropout_p
