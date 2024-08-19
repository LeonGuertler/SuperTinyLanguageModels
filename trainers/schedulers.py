"""Various Schedulers for Learning Rate and Dropout"""

import enum
import math

import pydantic
import torch.nn as nn
from pydantic import NonNegativeInt, PositiveFloat
from typing_extensions import Annotated


class LRSchedulerNames(str, enum.Enum):
    """Possible types of LR Scheduler"""

    CONSTANT = "constant"
    COSINE = "cosine"


class LRSchedulerConfig(pydantic.BaseModel):
    """Learning Rate Scheduler Configuration"""

    lr_scheduler_type: LRSchedulerNames.CONSTANT
    lr: PositiveFloat


class LRScheduler:
    """Constant LR scheduler"""

    def __init__(self, lr_scheduler_cfg: LRSchedulerConfig):
        self.lr = lr_scheduler_cfg.lr

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


class CosineLRSchedulerConfig(LRSchedulerConfig):
    """Cosine LR Scheduler Configuration"""

    lr_scheduler_type: LRSchedulerNames.COSINE
    warmup_iters: NonNegativeInt
    decay_iters: NonNegativeInt
    lr: PositiveFloat
    min_lr: PositiveFloat


class CosineLRScheduler(LRScheduler):
    """Basic Cosine LR scheduler with warmup and decay."""

    def __init__(self, lr_scheduler_cfg: CosineLRSchedulerConfig):
        """Initialize the scheduler"""
        super().__init__(lr_scheduler_cfg)
        self.warmup_iters = lr_scheduler_cfg.warmup_iters
        self.decay_iters = lr_scheduler_cfg.decay_iters
        self.lr = lr_scheduler_cfg.lr
        self.min_lr = lr_scheduler_cfg.min_lr

    def get_lr(self, iter_num):
        """Get the learning rate for the iteration number"""
        if iter_num < self.warmup_iters:
            return self.lr * iter_num / self.warmup_iters
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + math.cos((iter_num - self.warmup_iters) / self.decay_iters * math.pi)
        )


class DropoutSchedulerNames(str, enum.Enum):
    """Possible types of dropout scheduler. See indidivual
    Types for documentation"""

    CONSTANT = "constant"
    LINEAR = "linear"
    TRIANGLE = "triangle"


ProbabilityFloat = Annotated[float, pydantic.Field(ge=0, lt=1)]


class DropoutSchedulerConfig(pydantic.BaseModel):
    """Dropout Scheduler Configuration"""

    dropout_type: DropoutSchedulerNames
    dropout: ProbabilityFloat


class DropoutScheduler:
    """Constant Dropout Scheduler"""

    def __init__(self, dropout_cfg: DropoutSchedulerConfig):
        self.dropout_p = dropout_cfg.dropout

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


class LinearDropoutSchedulerConfig(DropoutSchedulerConfig):
    """Linear Dropout Scheduler Configuration
    Linearly moves between start_dropout_p and end_dropout_p between
    start_iter and end_iter. Has the value of start_dropout_p before
    and end_dropout_p after"""

    dropout_type: DropoutSchedulerNames.LINEAR
    start_iter: NonNegativeInt
    end_iter: NonNegativeInt
    start_dropout_p: ProbabilityFloat
    end_dropout_p: ProbabilityFloat


class LinearDropoutScheduler(DropoutScheduler):
    """Dropout Scheduler"""

    def __init__(self, dropout_cfg: LinearDropoutSchedulerConfig):
        """Initialize the dropout schedule"""
        super().__init__(dropout_cfg)
        self.start_iter = dropout_cfg.start_iter
        self.end_iter = dropout_cfg.end_iter
        self.start_dropout_p = dropout_cfg.start_dropout_p
        self.end_dropout_p = dropout_cfg.end_dropout_p

    def get_dropout(self, iter_num):
        """Return Constant Dropout"""
        if iter_num < self.start_iter:
            return self.start_dropout_p
        if iter_num >= self.end_iter:
            return self.end_dropout_p
        return self.start_dropout_p + (iter_num - self.start_iter) * (
            self.end_dropout_p - self.start_dropout_p
        ) / (self.end_iter - self.start_iter)


class TriangleDropoutSchedulerConfig(DropoutSchedulerConfig):
    """
    Args:
            dropout_trough: The minimum dropout probability
            dropout_peak: The maximum dropout probability
            num_iterations: The total number of iterations
            num_cycles: The number of cycles
    """

    dropout_type: DropoutSchedulerNames.TRIANGLE
    dropout_trough: ProbabilityFloat
    dropout_peak: ProbabilityFloat
    num_iterations: NonNegativeInt = 30000
    num_cycles: NonNegativeInt = 3


class TriangleDropoutScheduler(DropoutScheduler):
    """Triangle Dropout Scheduler. Ref: https://arxiv.org/pdf/1506.01186"""

    def __init__(self, dropout_cfg: TriangleDropoutSchedulerConfig):
        """Initialize the dropout schedule"""
        super().__init__(dropout_cfg)
        self.dropout_trough = dropout_cfg.dropout_trough
        self.dropout_peak = dropout_cfg.dropout_peak
        self.total_iterations = dropout_cfg.num_iterations
        self.cycle_length = self.total_iterations // dropout_cfg.num_cycles

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
