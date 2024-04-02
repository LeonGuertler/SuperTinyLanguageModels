"""Trainer class for training models with Next Token Prediction"""

import time
from contextlib import nullcontext

import torch
from models.build_models import build_model
from trainers import loss_fn as loss_fns
from trainers import scheduler as schedulers
from trainers import utilities


class BaseTrainer:
    """Base Trainer Class

    Uses subcomponents: optimizer, scheduler,
    model, dataloader, loss functions, logger"""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataloaders,
        loss_fn,
        config,
        log_fn=print,
        output_dir=".",
        device="cuda",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler: schedulers.CosineScheduler = scheduler
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        self.scaler = None
        self.eval_interval = config.training.eval_interval
        self.log_interval = config.training.log_interval
        self.checkpoint_interval = config.training.checkpoint_interval
        self.log_fn = log_fn
        self.output_dir = output_dir
        self.device = device
        self.max_iters = config.training.max_iters
        self.config = config

    @torch.no_grad()
    def estimate_loss(self, model, ctx, eval_iters=1000):
        """Estimate the loss"""
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters, device=ctx.device)
            for i in range(eval_iters):
                x, y = next(self.dataloaders(split))
                with ctx:
                    output = model(x)
                    losses[i] = self.loss_fn(output, y)
            out[split] = losses.mean().item()
        model.train()
        return out

    def run_step(self, ctx):
        """Run a single step of training"""
        for _ in range(self.gradient_accumulation_steps):
            x, y = next(self.dataloaders("train"))
            with ctx:
                output = self.model(x)
                loss = self.loss_fn(output, y)
                loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        if self.optimizer.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.optimizer.grad_clip,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)
        return loss

    def setup_ctx(self):
        """Get the context manager"""
        if self.device == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            ctx = nullcontext()
        return ctx

    def setup_scaler(self, ctx):
        """Setup the scaler"""
        self.scaler = torch.cuda.amp.GradScaler(enabled=ctx.dtype == torch.float16)

    def run_training_loop(self, ctx):
        """Run the training loop"""
        for iter_num in range(self.max_iters):
            t0 = time.time()
            lr = self.scheduler.get_lr()
            self.scheduler.apply_lr(self.optimizer, lr)
            # estimate the loss on the train/val sets
            if not iter_num % self.eval_interval:
                losses = self.estimate_loss(self.model, ctx)
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                self.log_fn(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                    }
                )
            # save checkpoints
            if not iter_num % self.checkpoint_interval:
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "iter_num": iter_num,
                    "config": self.config,
                }
                print(f"saving checkpoint to {self.output_dir}")
                torch.save(checkpoint, f"ckpt_{iter_num}.pt")

            loss = self.run_step(ctx)
            t1 = time.time()
            if not iter_num % self.log_interval:
                lossf = loss.item() * self.gradient_accumulation_steps
                print(
                    f"step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {t1-t0:.1f}s"
                )

    def train(self, seed=42):
        """Train the model"""
        utilities.set_seed(seed)
        ctx = self.setup_ctx()
        self.setup_scaler(ctx)
        self.run_training_loop(ctx)


def build_logger(config):
    """Build the logger"""
    if config.logger == "wandb":
        import wandb

        return wandb.log
    return print


def build_dataloader(model, split):
    """TODO: Replace this..."""
    model.tokenizer.prepare_data()
    for x, y in model.tokenizer.get_dataloader(split):
        yield x, y


def build_trainer(config):
    """Build the trainer"""
    model = build_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config.training.optimizer,
    )
    scheduler = schedulers.CosineScheduler(
        optimizer,
        **config.training.scheduler,
    )
    loss_fn = loss_fns.build_loss_fn(config.loss_fn)
    log_fn = build_logger(config)
    train_dl = build_dataloader(model, "train")
    val_dl = build_dataloader(model, "val")

    return BaseTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders={
            "train": train_dl,
            "val": val_dl,
        },
        log_fn=log_fn,
        loss_fn=loss_fn,
        config=config,
    )
