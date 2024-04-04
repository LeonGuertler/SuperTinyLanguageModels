"""Trainer class for training models with Next Token Prediction"""

import time
from contextlib import nullcontext

import torch
import wandb
from models.build_models import build_model
from trainers import utils


class BaseTrainer:
    """Base Trainer Class

    Uses subcomponents: optimizer, scheduler,
    model, dataloader, loss functions, logger"""

    def __init__(self, cfg, model, optimizer, scheduler, dataloader, loss_fn) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
        self.scaler = None
        self.eval_interval = cfg.training.eval_interval
        self.use_wandb = cfg.general.logging.wandb_log
        self.log_interval = cfg.training.log_interval
        self.checkpoint_interval = cfg.training.checkpoint_interval
        self.output_dir = cfg.training.output_dir
        self.max_iters = cfg.training.max_iters
        # For training, always force the device to be cuda
        assert torch.cuda.is_available(), "CUDA must be available for training"
        self.device = torch.device("cuda")
        self.ctx = self._setup_ctx()
        self.is_processed = False

    def _setup_ctx(self):
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

    def preprocess_data(self):
        """
        Preprocess the data
        """
        print("Preprocessing the training data")
        self.dataloader.prepare_data(tokenizer=self.model.embedder.tokenizer)
        self.is_processed = True

    @torch.no_grad()
    def estimate_loss(self, model, eval_iters=1000):
        """Estimate the loss"""
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = next(self.dataloader(split))
                with self.ctx:
                    output = model(x)
                    losses[i] = self.loss_fn(output, y)
            out[split] = losses.mean().item()
        model.train()
        return out

    def _run_step(self):
        """Run a single step of training"""
        for _ in range(self.gradient_accumulation_steps):
            x, y = next(self.dataloader("train"))
            with self.ctx:
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

    def setup_scaler(self):
        """Setup the scaler"""
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.ctx.dtype == torch.float16)

    def run_training_loop(self):
        """Run the training loop"""
        for iter_num in range(self.max_iters):
            t0 = time.time()
            lr = self.scheduler.get_lr()
            self.scheduler.apply_lr(self.optimizer, lr)
            # estimate the loss on the train/val sets
            if not iter_num % self.eval_interval:
                losses = self.estimate_loss(self.model)
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if self.use_wandb:
                    wandb.log(
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
                    "config": self.cfg,
                }
                print(f"saving checkpoint to {self.output_dir}")
                torch.save(checkpoint, f"ckpt_{iter_num}.pt")

            loss = self._run_step()
            t1 = time.time()
            if not iter_num % self.log_interval:
                lossf = loss.item() * self.gradient_accumulation_steps
                print(
                    f"step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {t1-t0:.1f}s"
                )

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.setup_scaler()
        self.run_training_loop()
