"""Learning Rate Scheduler"""

import math


class CosineScheduler:
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