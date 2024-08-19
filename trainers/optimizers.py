"""
A collection of optimizers used for training.
"""

import enum
import inspect

import pydantic
import torch


class OptimizerTypeNames(str, enum.Enum):
    """Possible types of Optimizers"""

    ADAMW = "AdamW"
    NANOGPT_ADAMW = "nanoGPTadamW"


class OptimizerConfig(pydantic.BaseModel):
    """
    Optimizer configuration
    """

    name: OptimizerTypeNames
    lr: float = 0.0006
    min_lr: float = 6.0e-05
    decay_lr: bool = True
    weight_decay: float | None = 0.1
    warmup_iters: int = 5000
    optimizer_type: str
    grad_clip: float = 1.0


class AdamWConfig(OptimizerConfig):
    """The nano gpt optimizer configuration"""

    name: OptimizerTypeNames = OptimizerTypeNames.ADAMW
    beta1: float = 0.9
    beta2: float = 0.95


class NanoGPTAdamWConfig(OptimizerConfig):
    """The nano gpt optimizer configuration"""

    name: OptimizerTypeNames = OptimizerTypeNames.NANOGPT_ADAMW
    beta1: float = 0.9
    beta2: float = 0.95


# pylint: disable=invalid-name
def configure_nanoGPT_optimizer(model, optimizer_cfg: AdamWConfig):
    """Configure the optimizer for NanoGPT"""
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": optimizer_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)},"
        f" with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)},"
        f" with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=optimizer_cfg.lr,
        betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
        **extra_args,
    )
    print(f"using fused AdamW: {use_fused}")

    return optimizer


# pylint: enable=invalid-name
