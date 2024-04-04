"""Trainer class for training models with Next Token Prediction"""

import time

import torch
import wandb
from omegaconf import OmegaConf
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
        self.gradient_accumulation_steps = (
            cfg.trainer.training.gradient_accumulation_steps
        )
        self.scaler = None
        self.use_wandb = cfg.general.logging.wandb_log
        self.checkpoint_dir = cfg.general.paths.checkpoint_dir
        # For training, always force the device to be cuda
        assert torch.cuda.is_available(), "CUDA must be available for training"
        self.ctx = self._setup_ctx()
        if self.use_wandb:
            self._setup_logging()


    def _setup_logging(self):
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
        grad_clip = self.cfg.trainer.optimizer.grad_clip
        if grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                grad_clip,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)
        return loss
    
    def _save_model(self, iter_num=0):
        """
        store the current model checkpoint.
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "config": self.cfg,
        }
        checkpoint_path = f"{self.checkpoint_dir}/ckpt_{iter_num}.pt"
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(
            checkpoint, 
            checkpoint_path
        )

    def run_training_loop(self):
        """Run the training loop"""
        for iter_num in range(self.cfg.trainer.training.max_iters):
            t0 = time.time()
            lr = self.scheduler.step(self.optimizer, iter_num)
            # estimate the loss on the train/val sets
            if not iter_num % self.cfg.trainer.training.eval_interval:
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
            if not iter_num % self.cfg.trainer.optimizer.checkpoint_interval:
                self._save_model(iter_num)


            loss = self._run_step()
            t1 = time.time()
            if not iter_num % self.cfg.trainer.training.log_interval:
                lossf = loss.item() * self.gradient_accumulation_steps
                print(
                    f"step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {t1-t0:.1f}s"
                )
        # save the final model
        self._save_model(iter_num)

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()


