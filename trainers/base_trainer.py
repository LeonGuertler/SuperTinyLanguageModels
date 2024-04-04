"""Trainer class for training models with Next Token Prediction"""

import time
from contextlib import nullcontext

from omegaconf import OmegaConf
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
        self.gradient_accumulation_steps = cfg.trainer.training.gradient_accumulation_steps
        self.scaler = None
        self.eval_interval = cfg.trainer.training.eval_interval
        self.use_wandb = cfg.general.logging.wandb_log
        self.log_interval = cfg.trainer.training.log_interval
        self.checkpoint_interval = cfg.trainer.optimizer.checkpoint_interval
        self.checkpoint_dir = cfg.general.paths.checkpoint_dir
        self.max_iters = cfg.trainer.training.max_iters
        self.grad_clip = cfg.trainer.optimizer.grad_clip
        # For training, always force the device to be cuda
        assert torch.cuda.is_available(), "CUDA must be available for training"
        self.device = torch.device("cuda")
        self.ctx = self._setup_ctx()
        self.is_processed = False
        self._setup_logging()

    def _setup_logging(self):
        if self.use_wandb:
            # set run name
            run_name = f"{self.cfg.model.model}_{self.cfg.trainer.dataset}_{self.cfg.model.tokenizer}"
            wandb.init(
                project=self.cfg.general.logging.wandb_project,
                config=OmegaConf.to_container(self.cfg),
                name=run_name,
            )
            wandb.init(project=self.cfg.general.logging.wandb_project)
            print("wand_b_initted")


    def _setup_ctx(self):
        """Get the context manager"""
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self._setup_scaler(dtype)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        return ctx

    def _setup_scaler(self, dtype=torch.float16):
        """Setup the scaler"""
        self.scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)

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
                x, y = self.dataloader.get_batch(split)
                with self.ctx:
                    output = model(x)
                    losses[i] = self.loss_fn(output, y)
            out[split] = losses.mean().item()
        model.train()
        return out

    def _run_step(self):
        """Run a single step of training"""
        for _ in range(self.gradient_accumulation_steps):
            x, y = self.dataloader.get_batch("train")
            with self.ctx:
                output = self.model(x)
                loss = self.loss_fn(output, y)
                loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)
        return loss

    def run_training_loop(self):
        """Run the training loop"""
        for iter_num in range(self.max_iters):
            t0 = time.time()
            lr = self.scheduler.get_lr(iter_num)
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
                print(f"saving checkpoint to {self.checkpoint_dir}")
                checkpoint_path = f"{self.checkpoint_dir}/ckpt_{iter_num}.pt"
                torch.save(
                    checkpoint,
                    checkpoint_path,
                )

            loss = self._run_step()
            t1 = time.time()
            if not iter_num % self.log_interval:
                lossf = loss.item() * self.gradient_accumulation_steps
                print(
                    f"step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {t1-t0:.1f}s"
                )
        # save the final model
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "config": self.cfg,
        }
        checkpoint_path = f"{self.checkpoint_dir}/final_checkpoint.pt"
        print(f"saving final checkpoint to {self.checkpoint_dir}")
        torch.save(checkpoint, checkpoint_path)

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()
