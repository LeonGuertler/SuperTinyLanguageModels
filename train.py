"""Training Script"""

import math
import time
from contextlib import nullcontext

import hydra.utils
import numpy as np
import torch
from models.build_models import build_model
from omegaconf import DictConfig, OmegaConf


@torch.no_grad()
def estimate_loss(model, eval_iters, ctx):
    """Estimate the loss over eval_iterations."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = model.get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Runs the learning rate schedule."""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


@hydra.main(config_path="configs/train/", config_name="baseline.yaml")
def main(model_cfg: DictConfig) -> None:
    """Run training from config."""
    # Load the general config file
    general_cfg_path = hydra.utils.to_absolute_path("configs/general_config.yaml")
    general_cfg = OmegaConf.load(general_cfg_path)

    # Merge the general configuration with the nanoGPT configuration
    cfg = OmegaConf.merge(general_cfg, model_cfg)

    # set the random seed
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # specify the device and dtype for training
    device = "cuda"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == "float16")

    tokens_pre_iteration = (
        cfg.training.gradient_accumulation_steps
        * cfg.training.batch_size
        * cfg.arch.context_window
    )
    print(f"Tokens per iteration: {tokens_pre_iteration}")

    iter_num = 0
    best_val_loss = 1e9

    # model
    model = build_model(config=cfg)
    model.to(device)

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay=cfg.training.optimizer.weight_decay,
        learning_rate=cfg.training.optimizer.lr,
        betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        device_type=device,
    )

    # start logging
    if cfg.logging.wandb_log:
        run_name = f"{cfg['arch']['model']}_{cfg['training']['dataset']}_{cfg['arch']['tokenizer']}"
        # pylint: disable=import-outside-toplevel
        import wandb

        wandb.init(
            project=cfg.logging.wandb_project,
            config=OmegaConf.to_container(cfg),
            name=run_name,
        )

    # start training
    x, y = model.get_batch()
    t0 = time.time()

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(
            iter_num,
            cfg.training.warmup_iters,
            cfg.training.lr_decay_iters,
            cfg.training.optimizer.lr,
            cfg.training.optimizer.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets
        if not iter_num % cfg.training.eval_interval:
            losses = estimate_loss(model, cfg.training.eval_iters, ctx)

            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if cfg.logging.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                    }
                )

        # save every 25 000 iterations
        if not iter_num % 25000:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": cfg,
            }
            print(f"saving checkpoint to {general_cfg.output_dir}")
            torch.save(checkpoint, f"ckpt_{iter_num}.pt")

        for _ in range(cfg.training.gradient_accumulation_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.training.gradient_accumulation_steps

            x, y = model.get_batch()
            scaler.scale(loss).backward()

        if cfg.training.optimizer.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.optimizer.grad_clip
            )

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush gradients asap
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if not iter_num % cfg.training.log_interval:
            lossf = loss.item() * cfg.training.gradient_accumulation_steps
            print(f"step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {dt:.1f}s")

        iter_num += 1

        if iter_num > cfg.training.max_iters:
            break

    # save the model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    print(f"saving checkpoint to {general_cfg.output_dir}")
    torch.save(checkpoint, "ckpt.pt")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
