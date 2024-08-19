"""Trainer class for training models with Next Token Prediction"""

import time
from contextlib import nullcontext

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function

from high_level_configs import GeneralConfig
from models import model_shell
from trainers import evaluation, utils
from trainers.config import TrainerConfig
from trainers.optimizers import OptimizerConfig, configure_nanoGPT_optimizer
from trainers.utils import aggregate_value, print_evaluation_results


# pylint: disable invalid-name
class BaseTrainer:
    """Base Trainer Class

    Uses subcomponents: optimizer, scheduler,
    model, dataloader, loss functions, logger"""

    def __init__(
        self,
        original_cfg: OmegaConf,
        general_cfg: GeneralConfig,
        training_cfg: TrainerConfig,
        model_cfg: model_shell.ModelShellConfig,
        model: model_shell.ModelShell,
        optimizer_cfg: OptimizerConfig,
        evaluation_cfg: evaluation.EvaluationConfig,
        train_dataloader,
        val_dataloader,
        loss_fn,
        gpu_id=None,
        lr_scheduler=None,
        dropout_scheduler=None,
    ) -> None:
        self.original_cfg = original_cfg
        self.general_cfg = general_cfg
        self.training_cfg = training_cfg
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.evaluation_cfg = evaluation_cfg
        self.model = model
        if gpu_id is not None:  # using ddp
            self.dist = True
            self.ddp_model = DDP(self.model, device_ids=[gpu_id])
        else:
            self.dist = False
            self.ddp_model = model
        self.gpu_id = gpu_id
        self.optimizer = configure_nanoGPT_optimizer(self.model, optimizer_cfg)
        self.lr_scheduler = lr_scheduler
        self.dropout_scheduler = dropout_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.gradient_accumulation_steps = (
            training_cfg.gradient_accumulation_steps
        )  # // torch.cuda.device_count() ## divide by number of GPUs to maximise throughput
        self.scaler = None
        self.use_wandb = general_cfg.logging.wandb_log
        self.checkpoint_dir = general_cfg.paths.checkpoint_dir
        self.cached_sets = {"train": {}, "val": {}}
        self.batch_size = training_cfg.batch_size  ## new

        # For training, always force the device to be cuda
        # assert torch.cuda.is_available(), "CUDA must be available for training"
        self.ctx = self._setup_ctx()
        if self.use_wandb and (
            self.gpu_id == 0 or not self.dist
        ):  ## ensures that only the first GPU logs to wandb
            self._setup_logging()
        if training_cfg.run_profiler and (
            self.gpu_id == 0 or not self.dist
        ):  ## ensures that only the first GPU runs the profiler
            self.run_profile()
            raise SystemExit

    def _setup_logging(self):
        # set run name
        run_name = (
            f"{self.model_cfg.model_shell_type}"
            f"_{self.model_cfg.core_model.core_model_type}"
            f"_{self.training_cfg.dataset}_{self.model_cfg.embedder.embedding_model_type}"
            f"_{self.model_cfg.vocab_size}"
        )
        wandb.init(
            project=self.general_cfg.logging.wandb_project,
            config=OmegaConf.to_container(self.original_cfg),
            name=run_name,
        )
        wandb.init(
            project=self.general_cfg.logging.wandb_project
        )  ## why does this happen twice???
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
            eval_iters = self.evaluation_cfg.eval_iters
        eval_results = {}
        self.model.eval()

        # eval on val set
        losses = []
        perplexities = []
        for i, (x, y) in enumerate(self.val_dataloader):
            x = x.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            y = y.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            with self.ctx:
                output, _ = self.model(x)

                # compute loss
                loss = self.loss_fn(output, y)
                losses.append(loss.item())

                # compute perplexity
                perplexity = torch.exp(
                    loss
                )  # since seq len is always the same during training anyway
                perplexities.append(perplexity.item())

            if i >= eval_iters:
                break

        avg_loss = aggregate_value(np.mean(losses), self.general_cfg.device)
        eval_results["Loss"] = avg_loss

        avg_perplexity = aggregate_value(np.mean(perplexities), self.general_cfg.device)
        eval_results["Perplexity"] = avg_perplexity

        evaluator_results = {}
        for evaluator_dict in self.evaluation_cfg.evaluators:
            evaluator = evaluation.get_evaluator_config(evaluator_dict)
            evaluator_results[evaluator.evaluator] = evaluation.train_eval(
                evaluator, self.model
            )
            # recurse over metrics to prepend the evaluator name as a prefix
            relabeled_results = {}
            for metric in evaluator_results[evaluator.evaluator]:
                relabeled_results[f"{evaluator.evaluator}/{metric}"] = (
                    evaluator_results[evaluator.evaluator][metric]
                )
            evaluator_results[evaluator.evaluator] = relabeled_results
        self.model.train()
        return eval_results, evaluator_results

    def _run_step(self):
        """Run a single step of training with gradient accumulation."""
        self.optimizer.zero_grad()  # Clear gradients at the start of accumulation

        for i, (x, y) in enumerate(self.train_dataloader):
            x = x.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            y = y.to(self.gpu_id if self.gpu_id is not None else self.model.device)

            # Enable or disable gradient synchronization based on the need for accumulation
            if self.dist and hasattr(self.ddp_model, "no_sync"):
                context_manager = (
                    self.ddp_model.no_sync()
                    if i != self.gradient_accumulation_steps - 1
                    else nullcontext()
                )
            else:
                context_manager = nullcontext()

            with context_manager:
                with self.ctx:  # Assuming self.ctx is something like torch.cuda.amp.autocast
                    output, aux_loss = self.ddp_model(x)
                    loss = self.loss_fn(
                        output, y
                    )  # + (aux_loss if aux_loss is not None else 0)
                    if aux_loss is not None:
                        loss += aux_loss
                # Scale loss to simulate larger effective batch size
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

            # Step and update only after accumulating enough gradients
            if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(
                self.train_dataloader
            ):
                if self.optimizer_cfg.grad_clip > 0:
                    # Unscale the gradients of the optimizer's assigned params in-place
                    self.scaler.unscale_(self.optimizer)
                    # Clip the gradients with normalization
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.optimizer_cfg.grad_clip
                    )

                # Perform a single optimization step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()  # Reset gradients after update

        return loss.item()  # Assuming loss is the average over the accumulated steps

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
                    self._run_step()  ## set the 'epoch' to ensure shuffle
                else:
                    with record_function("_run_step"):
                        self._run_step()  ## set the 'epoch' to ensure shuffle
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
            "config": self.original_cfg,
        }
        checkpoint_path = f"{self.checkpoint_dir}/ckpt_{iter_num}.pt"
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def run_training_loop(self):
        """Run the training loop"""
        for iter_num in range(self.training_cfg.max_iters):
            start_time = time.time()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(self.optimizer, iter_num)
            else:
                lr = self.optimizer.param_groups[0]["lr"]
            dropout = self.dropout_scheduler.step(self.model, iter_num)
            # estimate the loss on the train/val sets
            if (
                not iter_num % self.evaluation_cfg.eval_interval
            ):  # run on first iter to prevent bugs causing it to crash
                eval_results, benchmark_results = self.estimate_performance()

                # print the evals as table
                # evals format is d1: type d2: train/val
                print_evaluation_results(
                    iter_num=iter_num,
                    eval_results=eval_results,
                    benchmark_results=benchmark_results,
                )

                # Log to wandb
                if (
                    self.gpu_id == 0 or self.gpu_id is None
                ) and self.use_wandb:  # ensure only the first GPU logs
                    log_dict = {"iter": iter_num, "lr": lr, "dropout": dropout}
                    log_dict.update(
                        eval_results
                    )  # Directly add evals to the log dictionary
                    log_dict.update(
                        {k: v for k, v in benchmark_results.items()}
                    )  # Add benchmark results to the log dictionary

                    wandb.log(log_dict)

            # save checkpoints
            if (
                not iter_num % self.training_cfg.checkpoint_interval
                and iter_num > 0
                and (self.gpu_id in (0, None))  ## ensure only the first GPU prints
            ):
                self._save_model(iter_num)

            loss = self._run_step()  ## set the 'epoch' to ensure shuffle
            end_time = time.time()
            if not iter_num % self.training_cfg.log_interval and iter_num > 0:
                lossf = loss * self.gradient_accumulation_steps

                ## aggregate the loss across all GPUs
                lossf = aggregate_value(lossf, self.general_cfg.device)

                ## print and log the result only on the first GPU after aggregation
                print(
                    f"All GPU(s): step {iter_num}: loss {lossf:.4f},"
                    f" lr {lr:.1e}, dt {end_time-start_time:.1f}s"
                )
                if (self.gpu_id == 0 or self.gpu_id is None) and self.use_wandb:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "loss": lossf,
                            "lr": lr,
                            "dropout": dropout,
                        }
                    )
        # save the final model
        if (
            self.gpu_id == 0 or self.gpu_id is None
        ):  ## ensure only the first GPU saves the model
            self._save_model(iter_num)

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()
