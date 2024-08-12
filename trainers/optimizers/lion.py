"""
A pytorch implementation of the LION optimizer 
https://arxiv.org/pdf/2302.06675
taken from
https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py
"""
import torch 
from typing import Tuple, Callable


def exists(val):
    return val is not None

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay
    p.data.mul_(1. - lr * wd)

    # weight update
    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1. - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient
    exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)


class Lion(torch.optim.optimizer.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn


    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state, decoupled_wd, init_lr = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p], self.decoupled_wd, self._init_lr

                # maybe decoupled weight decay

                if decoupled_wd:
                    wd /= init_lr

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss