"""Trainer class for training models with Next Token Prediction"""

import time

import torch
import wandb
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function

from models import model_shell
from trainers import dataloader as train_dataloader
from trainers import utils

from trainers.loss_fn import (
    compute_perplexity,
)


# pylint: disable invalid-name
class BaseTrainer:
    """Base Trainer Class

    Uses subcomponents: optimizer, scheduler,
    model, dataloader, loss functions, logger"""

    def __init__(
        self,
        cfg,
        model: model_shell.ModelShell,
        optimizer,
        dataloader: train_dataloader.BaseDataloader,
        loss_fn,
        lr_scheduler=None,
        dropout_scheduler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dropout_scheduler = dropout_scheduler
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.gradient_accumulation_steps = cfg["trainer"]["training"][
            "gradient_accumulation_steps"
        ]
        self.scaler = None
        self.use_wandb = cfg["general"]["logging"]["wandb_log"]
        self.checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]
        self.cached_sets = {"train": {}, "val": {}}

        # For training, always force the device to be cuda
        assert torch.cuda.is_available(), "CUDA must be available for training"
        self.ctx = self._setup_ctx()
        if self.use_wandb:
            self._setup_logging()
        if cfg.trainer.training.run_profiler:
            self.run_profile()
            raise SystemExit

    def _setup_logging(self):
        # set run name
        run_name = (
            f"{self.cfg.model['model_shell_type']}"
            f"_{self.cfg.model['core_model']['core_model_type']}"
            f"_{self.cfg.trainer['dataset']}_{self.cfg.model['embedder']['embedding_model_type']}"
            f"_{self.cfg.model['vocab_size']}"
        )
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
        self.dataloader.prepare_data()

    @torch.no_grad()
    def estimate_performance(self, eval_iters=None):
        """Estimate the loss"""
        if eval_iters is None:
            eval_iters = self.cfg.trainer.training.eval_iters
        loss = {}
        perplexity = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            perplexities = torch.zeros(eval_iters)
            for i in range(eval_iters):
                # use cached eval if available

                if i in self.cached_sets[split]:
                    print("use cached test set")
                    x = self.cached_sets[split][i]["x"]
                    y = self.cached_sets[split][i]["y"]
                    token_lengths = self.cached_sets[split][i]["token_lengths"]
                    char_lengths = self.cached_sets[split][i]["char_lengths"]
                    mask = self.cached_sets[split][i]["mask"]
                else:
                    print("process test set")
                    x, y = self.dataloader.get_batch(split)
                    token_lengths, char_lengths, mask = self.model.embedding_model.get_sequence_info(x)
                    self.cached_sets[split][i] = {
                        "x": x,
                        "y": y,
                        "token_lengths": token_lengths,
                        "char_lengths": char_lengths,
                        "mask": mask,
                    }
                with self.ctx:
                    output, _ = self.model(x)
                    losses[i] = self.loss_fn(output, y, mask=mask)
                    perplexities[i] = compute_perplexity(
                        logits=output,
                        y=y,
                        token_lengths=token_lengths,
                        char_lengths=char_lengths,
                        mask=mask,
                    )
            loss[split] = losses.mean().item()
            perplexity[split] = perplexities.mean().item()
        self.model.train()
        return loss, perplexity

    def _run_step(self):
        """Run a single step of training"""
        for _ in range(self.gradient_accumulation_steps):
            x, y = self.dataloader.get_batch("train")
            with self.ctx:
                output, aux_loss = self.model(x)
                loss = self.loss_fn(output, y)
                if aux_loss is not None:
                    loss += aux_loss
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

    def run_profile(self):
        """Run the profiler"""
        utils.profilize(self.model)
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i in range(10):
                if i <= 3:
                    self._run_step()
                else:
                    with record_function("_run_step"):
                        self._run_step()
            # place profile in dictionary
        backwards_prof = prof.key_averages().table(sort_by="self_cpu_time_total")
        print(backwards_prof)
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            self.estimate_performance(eval_iters=1)
            with record_function("estimate_performance"):
                self.estimate_performance(eval_iters=10)
            # place profile in dictionary
        forwards_prof = prof.key_averages().table(sort_by="self_cpu_time_total")
        print(forwards_prof)

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
        torch.save(checkpoint, checkpoint_path)

    def run_training_loop(self):
        """Run the training loop"""
        for iter_num in range(self.cfg.trainer.training.max_iters):
            start_time = time.time()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(self.optimizer, iter_num)
            else:
                lr = self.optimizer.param_groups[0]["lr"]
            dropout = self.dropout_scheduler.step(self.model, iter_num)
            # estimate the loss on the train/val sets
            if (not iter_num % self.cfg.trainer.training.eval_interval) and iter_num > 0:
                s0 = time.time()
                losses, perplexities = self.estimate_performance()
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f},"
                    f" val loss {losses['val']:.4f}, dt {time.time()-s0:.1f}s"
                )
                print(
                    f"step {iter_num}: train perplexity {perplexities['train']:.4f},"
                    f" val perplexity {perplexities['val']:.4f}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "dropout": dropout,
                            "train/perplexity": perplexities["train"],
                            "val/perplexity": perplexities["val"],
                        }
                    )
            # save checkpoints
            if (
                not iter_num % self.cfg.trainer.training.checkpoint_interval
                and iter_num > 0
            ):
                self._save_model(iter_num)

            loss = self._run_step()
            end_time = time.time()
            if not iter_num % self.cfg.trainer.training.log_interval and iter_num > 0:
                lossf = loss.item() * self.gradient_accumulation_steps
                print(
                    f"step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {end_time-start_time:.1f}s"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "loss": lossf,
                            "lr": lr,
                            "dropout": dropout,
                        }
                    )
        # save the final model
        self._save_model(iter_num)

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()
