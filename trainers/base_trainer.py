"""Trainer class for training models with Next Token Prediction"""

import time

import torch
import wandb
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function
from copy import deepcopy
from contextlib import nullcontext

from models import model_shell
from trainers import dataloader as train_dataloader
from trainers import utils

from trainers.loss_fn import (
    compute_perplexity
)
from trainers.evaluator import train_eval

import numpy as np
from itertools import islice
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from trainers.utils import aggregate_value, get_qk_scores, get_prenormalized_attention_list, project_student_to_teacher_hs, project_student_to_teacher_emb


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
        gpu_id, 
        lr_scheduler=None,
        dropout_scheduler=None,
        projection=None,
        teacher_model=None,
    ) -> None:
        self.model = model
        self.DDP_model = DDP(self.model, device_ids=[gpu_id])
        self.gpu_id = gpu_id 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dropout_scheduler = dropout_scheduler
        self.dataloader = dataloader
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
        if self.use_wandb and self.gpu_id == 0: ## ensures that only the first GPU logs to wandb
            self._setup_logging()
        if cfg.trainer.training.run_profiler and self.gpu_id == 0: ## ensures that only the first GPU runs the profiler
            self.run_profile()
            raise SystemExit
        
        ## For knowledge distillation, set the teacher model and hyperparameters
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.perform_kd = True
            self.kd_cfg = utils.init_kd_cfg(cfg)
            self.projection = projection
            
            ## import the distillation loss function
            from trainers.loss_fn import distillation_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
        else:
            self.perform_kd = False

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

    def _get_dataloader(self, split):
        """
        Feeds dataloader into PyTorch's DataLoader.
        """
        ## return the dataloader if it has already been cached
        if split in self.train_val_dataloaders:
            return self.train_val_dataloaders[split]
        
        ## if the dataloader has not been created, create it
        # set the split data
        dataset = self.dataloader.split_dataloader(split)
        # create the dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 0,
        )

        ## cache the dataloader
        self.train_val_dataloaders[split] = dataloader

        return dataloader

    @torch.no_grad()
    def estimate_performance(self, eval_iters=None):
        """Estimate the loss"""
        if eval_iters is None:
            eval_iters = self.cfg.trainer.training.eval_iters
        loss = {}
        perplexity = {}
        self.model.eval()
        for split in ["train", "val"]:

            ## initialize the loss, perplexity
            losses = torch.zeros(eval_iters)
            perplexities = torch.zeros(eval_iters)
            
            ## initialize Pytorch's DataLoader
            dataloader = self._get_dataloader(split)

            for i, (x, y) in enumerate(dataloader):
                if i > eval_iters - 1:
                    break
                # use cached eval if available
                if i in self.cached_sets[split]:
                    print("use cached test set")
                    x = self.cached_sets[split][i]["x"]
                    y = self.cached_sets[split][i]["y"]
                    char_lengths = self.cached_sets[split][i]["char_lengths"]
                    mask = self.cached_sets[split][i]["mask"]
                else:
                    print("process test set")
                    (
                        char_lengths,
                        mask,
                    ) = self.model.embedding_model.get_sequence_info(x)
                    self.cached_sets[split][i] = {
                        "x": x,
                        "y": y,
                        "char_lengths": char_lengths,
                        "mask": mask,
                    }
                with self.ctx:
                    output, _ = self.DDP_model(x)
                    losses[i] = self.loss_fn(output, y, mask=mask)
                    perplexities[i] = compute_perplexity(
                        logits=output,
                        y=y,
                        char_lengths=char_lengths,
                        mask=mask,
                    )

            ## aggregate the loss and perplexity across all GPUs
            avg_loss = aggregate_value(losses.mean().item(), self.cfg.general.device)
            loss[split] = avg_loss
            avg_perplexity = aggregate_value(perplexities.mean().item(), self.cfg.general.device)
            perplexity[split] = avg_perplexity

        evaluator_results = {}
        for evaluator in self.cfg.trainer["eval"]:
            evaluator_results[evaluator["evaluator"]] = train_eval(evaluator, self.model)
            # recurse over metrics to prepend the evaluator name as a prefix
            relabeled_results = {}
            for metric in evaluator_results[evaluator["evaluator"]]:
                relabeled_results[f"{evaluator['evaluator']}/{metric}"] = evaluator_results[evaluator["evaluator"]][metric]
            evaluator_results[evaluator["evaluator"]] = relabeled_results
        self.model.train()
        return loss, perplexity, evaluator_results

    def _run_step(self):
        """Run a single step of training"""
        ## init Pytorch's DataLoader
        dataloader = self._get_dataloader("train")
        for iter, (x, y) in enumerate(dataloader):
            if iter != self.gradient_accumulation_steps - 1:
                ddp_no_sync_ctx = self.DDP_model.no_sync()
            else:
                ddp_no_sync_ctx = nullcontext()
            with ddp_no_sync_ctx:
                with self.ctx:
                    if self.perform_kd:
                        ## get the teacher model output, without gradient calculation
                        with torch.no_grad():
                            teacher_output, _ = self.teacher_model(x)
                            
                            # teacher_QKs = get_qk_scores(self.teacher_model, x)
                            # teacher_attns = get_prenormalized_attention_list(teacher_QKs, teacher_model=self.teacher_model)
                            # teacher_hidden_states = self.teacher_model.core_model.hidden_states[1:]
                            # teacher_embeddings = self.teacher_model.embedding_model.embedding_output
                            # print(f"Teacher embeddings same: {teacher_embeddings_a == teacher_embeddings_b}")

                        ## get the student model output, with gradient calculation
                        student_output, aux_loss = self.DDP_model(x)

                        # student_QKs = self.DDP_model.module.core_model.qk_lists
                        # student_attns = get_prenormalized_attention_list(student_QKs)
                        # student_hidden_states = self.DDP_model.module.core_model.hidden_states
                        # student_embeddings = self.DDP_model.module.embedding_model.embedding_output
                        # print(f"Student embeddings shape: {student_embeddings.shape}")
                        
                        # ## take every k-th attention matrix from teacher to match all students
                        # k = len(teacher_attns) // len(student_attns)
                        # teacher_attns = teacher_attns[::k] # get every k-th attention matrix
                        # teacher_hidden_states = teacher_hidden_states[::k] # get every k-th hidden state

                        # ## calculate the transformer's attention and hidden_state loss over each layer
                        # attn_loss = 0.0
                        # hs_loss = 0.0
                        # for i in range(len(student_attns)):

                        #     ## project the student's attention and hidden states to the teacher's dimensions
                        #     student_hidden_states[i] = project_student_to_teacher_hs(self.projection, student_hidden_states[i])

                        #     mean_student_attns = torch.mean(student_attns[i], dim=1) ## mean over the heads
                        #     mean_teacher_attns = torch.mean(teacher_attns[i], dim=1) ## mean over the heads
                        #     attn_loss += torch.nn.functional.mse_loss(mean_student_attns, mean_teacher_attns)
                            
                        #     ## calculate the mean squared error loss between the student and teacher's attention and hidden states
                        #     hs_loss += torch.nn.functional.mse_loss(student_hidden_states[i], teacher_hidden_states[i])

                        ## calculate the embeddings loss
                        # student_embeddings = project_student_to_teacher_emb(self.projection, student_embeddings)
                        # embeddings_loss = torch.nn.functional.mse_loss(student_embeddings, teacher_embeddings)

                        ## calculate the soft_targets loss
                        soft_targets_loss = self.distillation_loss_fn(student_output, teacher_output, self.kd_cfg['temperature'])

                        ## calculate the cross entropy label loss
                        label_loss = self.loss_fn(student_output, y)
                        
                        ## uncomment to check label and soft targets loss
                        # print(f"Soft Targets Loss: {soft_targets_loss.item()}")
                        # print(f"Cross Entropy Loss: {label_loss.item()}")

                        ## combine the two losses
                        # loss = self.kd_cfg['embedding_loss_weight']*embeddings_loss + self.kd_cfg['attn_loss_weight']*attn_loss + self.kd_cfg['hs_loss_weight']*hs_loss + self.kd_cfg['soft_targets_loss_weight']*soft_targets_loss + self.kd_cfg['label_loss_weight']*label_loss
                        loss = self.kd_cfg['soft_targets_loss_weight']*soft_targets_loss + (1.0 - self.kd_cfg['soft_targets_loss_weight'])*label_loss

                    
                    else:
                        output, aux_loss = self.DDP_model(x)
                        loss = self.loss_fn(output, y)
                    
                    if aux_loss is not None:
                        loss += aux_loss
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
            if iter == self.gradient_accumulation_steps - 1:
                break

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
                    self._run_step() ## set the 'epoch' to ensure shuffle
                else:
                    with record_function("_run_step"):
                        self._run_step() ## set the 'epoch' to ensure shuffle
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

            # # Save initial weights of the projection layer
            # initial_weights = self.projection.projection_hs.weight.clone().detach()

            loss = self._run_step() ## set the 'epoch' to ensure shuffle

            # # Check if weights have been updated
            # updated_weights = self.projection.projection_hs.weight.clone().detach()

            # Compare initial and updated weights
            # weights_changed = not torch.equal(initial_weights, updated_weights)
            # print(f"Projection layer weights changed: {weights_changed}")

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
