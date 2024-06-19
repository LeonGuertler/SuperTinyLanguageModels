"""Trainer class for training models with Next Token Prediction"""

import time

import torch
import wandb
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function
from copy import deepcopy
from contextlib import nullcontext

from models import model_shell
from trainers import datasets as train_dataloader
from trainers import utils

from trainers.loss_fn import (
    compute_perplexity
)
from trainers.evaluator import train_eval

import numpy as np
from itertools import islice
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler
from trainers.utils import aggregate_value


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
        train_dataloader,
        val_dataloader,
        loss_fn,
        gpu_id=None, 
        lr_scheduler=None,
        dropout_scheduler=None,
    ) -> None:
        self.model = model
        if gpu_id is not None: # using ddp
            self.dist = True
            self.DDP_model = DDP(self.model, device_ids=[gpu_id])
        else:
            self.dist = False
            self.DDP_model = model
        self.gpu_id = gpu_id 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dropout_scheduler = dropout_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_val_dataloaders = {}
        self.loss_fn = loss_fn
        self.cfg = cfg
        assert self.cfg["trainer"]["training"]["gradient_accumulation_steps"] % torch.cuda.device_count() == 0, "Gradient Accumulation Steps must be divisible by the number of GPUs"
        self.gradient_accumulation_steps = cfg["trainer"]["training"][
            "gradient_accumulation_steps"
        ] // torch.cuda.device_count() ## divide by number of GPUs to maximise throughput
        self.scaler = None
        self.use_wandb = cfg["general"]["logging"]["wandb_log"]
        self.checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]
        self.cached_sets = {"train": {}, "val": {}}
        self.batch_size = cfg["trainer"]["training"]["batch_size"] ## new

        # For training, always force the device to be cuda
        assert torch.cuda.is_available(), "CUDA must be available for training"
        self.ctx = self._setup_ctx()
        if self.use_wandb and (self.gpu_id == 0 or not self.dist): ## ensures that only the first GPU logs to wandb
            self._setup_logging()
        if cfg.trainer.training.run_profiler and (self.gpu_id == 0 or not self.dist): ## ensures that only the first GPU runs the profiler
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


    @torch.no_grad()
    def estimate_performance(self, eval_iters=None):
        """Estimate the loss"""
        if eval_iters is None:
            eval_iters = self.cfg.trainer.training.eval_iters
        loss = {}
        perplexity = {}
        divergency = {}
        self.model.eval()

        # eval on val set 
        losses = []
        perplexities = []
        for i, (X, y) in enumerate(self.val_dataloader):
            with self.ctx:
                output, _ = self.model(X)
                loss = self.loss_fn(output, y)
                losses.append(loss.item())
                perplexities.append(
                    compute_perplexity(
                        logits=output,
                        y=y,
                    )
                )
            if i >= eval_iters:
                break
        
        # aggregate the loss and perplexity across all GPUs
        avg_loss = aggregate_value(np.mean(losses), self.cfg.general.device)
        loss["val"] = avg_loss
        avg_perplexity = aggregate_value(np.mean(perplexities), self.cfg.general.device)
        perplexity["val"] = avg_perplexity

        evaluator_results = {}
        for evaluator in self.cfg.trainer["eval"]:
            evaluator_results[evaluator["evaluator"]] = train_eval(evaluator, self.model.module)
            # recurse over metrics to prepend the evaluator name as a prefix
            relabeled_results = {}
            for metric in evaluator_results[evaluator["evaluator"]]:
                relabeled_results[f"{evaluator['evaluator']}/{metric}"] = evaluator_results[evaluator["evaluator"]][metric]
            evaluator_results[evaluator["evaluator"]] = relabeled_results
        self.model.train()
        return loss, perplexity, evaluator_results




    def _run_step(self, epoch = 0):
        """Run a single step of training"""
        for iter, (x, y) in enumerate(islice(self.train_dataloader, self.gradient_accumulation_steps)):
            # push x,y to device
            # if gpu_id is not none, push to it
            if self.gpu_id is not None:
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)
            else:
                # push to model device
                x = x.to(self.model.device)
                y = y.to(self.model.device)

            with self.ctx:
                output, aux_loss = self.DDP_model(x)
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
                    self._run_step(i) ## set the 'epoch' to ensure shuffle
                else:
                    with record_function("_run_step"):
                        self._run_step(i) ## set the 'epoch' to ensure shuffle
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
            "model": self.model.module.state_dict(), ## added the 'module' attribute (https://discuss.pytorch.org/t/access-to-attributes-of-model-wrapped-in-ddp/130572/2)
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
            if (
                not iter_num % self.cfg.trainer.training.eval_interval
            ) and iter_num > 0:
                s0 = time.time()
                losses, perplexities, benchmark_results = self.estimate_performance()
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f},"
                    f" val loss {losses['val']:.4f}, dt {time.time()-s0:.1f}s"
                )
                print(
                    f"step {iter_num}: train perplexity {perplexities['train']:.4f},"
                    f" val perplexity {perplexities['val']:.4f}"
                )
                print(
                    f"step {iter_num}: benchmark results {benchmark_results}"
                )

                if self.gpu_id == 0: ## ensure only the first GPU logs
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
                                **{
                                    f"benchmark/{k}": v
                                    for k, v in benchmark_results.items()
                                },
                            }
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
                            **{
                                k: v
                                for k, v in benchmark_results.items()
                            },
                        }
                    )
            # save checkpoints
            if (
                not iter_num % self.cfg.trainer.training.checkpoint_interval
                and iter_num > 0
                and self.gpu_id == 0 ## ensure only the first GPU prints
            ):
                self._save_model(iter_num)

            loss = self._run_step(iter_num) ## set the 'epoch' to ensure shuffle
            end_time = time.time()
            if not iter_num % self.cfg.trainer.training.log_interval and iter_num > 0:
                lossf = loss.item() * self.gradient_accumulation_steps

                ## uncomment the following line to print the loss on all GPUs
                # print(f"GPU {self.gpu_id}: step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {end_time-start_time:.1f}s")

                ## aggregate the loss across all GPUs
                lossf = aggregate_value(lossf, self.cfg.general.device)

                ## print and log the result only on the first GPU after aggregation
                print(f"All GPU(s): step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {end_time-start_time:.1f}s")
                if self.gpu_id == 0 and self.use_wandb:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "loss": lossf,
                            "lr": lr,
                            "dropout": dropout,
                        }
                    )
        # save the final model
        if self.gpu_id == 0: ## ensure only the first GPU saves the model
            self._save_model(iter_num)

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()

