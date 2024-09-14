"""Trainer class for training models with Next Token Prediction"""

import time
import wandb
from omegaconf import OmegaConf
from contextlib import nullcontext

# local imports
from trainers import utils, data_utils
from trainers.evaluator import train_eval_mcq, train_eval_text_modeling
from trainers.utils import aggregate_value, print_evaluation_results
from models.utils import print_model_stats

import numpy as np
from itertools import islice
import torch
from torch.nn.parallel import DistributedDataParallel as DDP 


# pylint: disable invalid-name
class BaseTrainer:
    """Base Trainer Class

    Uses subcomponents: optimizer, scheduler,
    model, dataloader, loss functions, logger"""

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        loss_fn,
        gpu_id=None, 
        lr_scheduler=None,
        loaded_train_config=None,
    ) -> None:
        self.model = model
        # print model stats and save them 
        total_params_formated = print_model_stats(model)

        if gpu_id is not None: # using ddp
            self.dist = True
            self.DDP_model = DDP(self.model, device_ids=[gpu_id])
        else:
            self.dist = False
            self.DDP_model = model

        self.gpu_id = gpu_id 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader_iter = iter(train_dataloader)
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.cfg = cfg

        # Load prev training parameters as necessary
        if loaded_train_config is not None:
            self.current_iter = loaded_train_config["iter_num"]

            if self.cfg["trainer"].get("load_prev_optimizer_state", False):
                print("Loading the previous optimizer state")
                self.optimizer.load_state_dict(loaded_train_config["optimizer"])

        else:
            self.current_iter = 0 


        # adjusting the correct batch-size accumulation step ratio for each node
        self.gradient_accumulation_steps = cfg["trainer"][
            "gradient_accumulation_steps"
        ] // torch.cuda.device_count() if torch.cuda.is_available() else cfg["trainer"][
            "gradient_accumulation_steps"
        ]## divide by number of GPUs to maximise throughput


        self.scaler = None
        self.ctx = self._setup_ctx()

        self.use_wandb = cfg["general"]["logging"]["wandb_log"]
        self.checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]
        self.batch_size = cfg["trainer"]["batch_size"] 
        self.evaluate_byte_metrics = self.cfg["trainer"]["eval"].get("eval_byte_metrics", False)


        if self.use_wandb and (self.gpu_id == 0 or not self.dist): ## ensures that only the first GPU logs to wandb
            self._setup_logging(
                total_parameter_count_str=total_params_formated
            )


    def _setup_logging(self, total_parameter_count_str=None):
        # check if run_name was provided
        if self.cfg["general"]["logging"].get("run_name", None) is not None:
            run_name = self.cfg["general"]["logging"]["run_name"] + f" (Size: {total_parameter_count_str})"
        else:
            # provide a generic (hopefully descriptive) run name if none was provided
            run_name = (
                f"{self.cfg.model['model_shell_type']}"
                f"_{self.cfg.model['embedding_model_type']}"
                f"_{self.cfg.model['core_model_type']}"
                f"_{self.cfg.model['lm_head_type']}"
                f"_{self.cfg.trainer['dataset']}"
                f"_{self.cfg.model['vocab_size']}"
                f"_Parameters_{total_parameter_count_str}"
            )

        wandb.init(
            project=self.cfg.general.logging.wandb_project, 
            config=OmegaConf.to_container(self.cfg),
            name=run_name,
        )
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
            eval_iters = self.cfg["trainer"]["eval_iters"]
        eval_results = {}
        self.model.eval()

        # initialize accumulators
        total_loss = 0
        total_bytes = 0
        total_token_count = 0 # For regular loass and perplexity

        # Initialize accumulators for byte-level metrics
        total_byte_loss = 0.0
        total_byte_log_probs = 0.0 # For perplexity calculation

        for i, (x, y) in enumerate(self.val_dataloader):
            x = x.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            y = y.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            with self.ctx:
                output, _ = self.model(x)
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    y.view(-1),
                    reduction='sum'
                )
                loss = self.loss_fn(output, y) # will average loss per token

            # Accumulate token-level metrics
            total_loss += loss.item()
            total_token_count += y.size(0)

            if self.evaluate_byte_metrics:
                # Compute byte counts for current batch
                y_tokens = y.view(-1).cpu().numpy()
                byte_counts = self.model.embedding_model.get_byte_lengths(y_tokens)
                batch_byte_count = byte_counts.sum()
                total_bytes += batch_byte_count

                # Accumulate byte-level loss
                total_byte_loss += loss_value/y.size(0) * batch_byte_count

            if i >= eval_iters:
                break
        
        # Calculate average metrics
        avg_loss = total_loss / total_token_count if total_token_count > 0 else float('inf')
        avg_perplexity = np.exp(avg_loss)



        # Store in eval_results
        eval_results["Val. Loss"] = aggregate_value(avg_loss, self.cfg.general.device)
        eval_results["Val. Perplexity"] = aggregate_value(avg_perplexity, self.cfg.general.device)

        if self.evaluate_byte_metrics:
            # Byte-normalized metrics
            byte_avg_loss = total_byte_loss / total_bytes if total_bytes > 0 else float('inf')
            byte_avg_perplexity = np.exp(byte_avg_loss) if byte_avg_loss < 100 else float('inf')  # Avoid overflow
            
            # Store byte-level metrics
            eval_results["Val. Loss (Bytes)"] = aggregate_value(byte_avg_loss, self.cfg.general.device)
            eval_results["Val. Perplexity (Bytes)"] = aggregate_value(byte_avg_perplexity, self.cfg.general.device)
        

        # get the mcq eval results
        eval_results.update(
            train_eval_mcq(
                model=self.model,
                num_samples=self.cfg["trainer"]["eval"].get("mcq_num_samples", None),
                benchmark_list=self.cfg["trainer"]["eval"].get("mcq_benchmarks", []),
            )
        )
        # get the text modeling eval results
        eval_results.update(
            train_eval_text_modeling(
                model=self.model,
                topic_list=self.cfg["trainer"]["eval"].get("text_modeling_topics", []),
                eval_dir=self.cfg["general"]["paths"]["eval_dir"],
            )
        )

        # set model back into train mode
        self.model.train()

        return eval_results


    def _run_step(self):
        """Run a single step of training with gradient accumulation."""
        self.optimizer.zero_grad()  # Clear gradients at the start of accumulation

        accumulated_loss = 0
        for i in range(self.gradient_accumulation_steps):
            # get the next batch
            x, y = next(self.train_dataloader_iter)
            x = x.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            y = y.to(self.gpu_id if self.gpu_id is not None else self.model.device)

            # Enable or disable gradient synchronization based on the need for accumulation
            if self.dist and hasattr(self.DDP_model, 'no_sync'):
                context_manager = self.DDP_model.no_sync() if i != self.gradient_accumulation_steps - 1 else nullcontext()
            else:
                context_manager = nullcontext()

            with context_manager:
                with self.ctx: 
                    output, aux_loss = self.DDP_model(x)
                    loss = self.loss_fn(output, y)
                    if aux_loss is not None:
                        loss += aux_loss

                # Scale loss to simulate larger effective batch size
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()

        # once graidents are accumulated, step 
        if self.cfg.trainer.optimizer.grad_clip > 0:
            # Unscale the gradients of the optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Clip the gradients with normalization
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.optimizer.grad_clip)
        
        # Perform a single optimization step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()  # Reset gradients after update

        return accumulated_loss


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
        print("Training loop is starting")
        for iter_num in range(self.current_iter, self.cfg["trainer"]["max_iters"]):
            start_time = time.time()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(self.optimizer, iter_num)
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            # estimate the loss on the train/val sets
            if (
                not iter_num % self.cfg["trainer"]["eval_interval"]
            ): # run on first iter to prevent bugs causing it to crash
                eval_results = self.estimate_performance()

                # print the evals as table
                # evals format is d1: type d2: train/val
                print_evaluation_results(
                    iter_num=iter_num, 
                    eval_results=eval_results, 
                )

                # extend eval results with general information
                eval_results.update(
                    {
                        "iter": iter_num,
                        "lr": lr,
                        "token_num": self.batch_size*self.gradient_accumulation_steps*iter_num*self.cfg.model["context_window"],                        
                    }
                )

                # Log to wandb
                if (self.gpu_id == 0 or self.gpu_id is None) and self.use_wandb:  # ensure only the first GPU logs
                    wandb.log(eval_results)

            # save checkpoints
            if (
                not iter_num % self.cfg["trainer"]["checkpoint_interval"]
                and iter_num > 0
                and (
                    self.gpu_id == 0
                    or self.gpu_id == None
                 ) ## ensure only the first GPU prints
            ):
                self._save_model(iter_num)


            lossf = self._run_step() 
            end_time = time.time()
            if not iter_num % self.cfg["trainer"]["log_interval"] and iter_num > 0:
                ## uncomment the following line to print the loss on all GPUs
                # print(f"GPU {self.gpu_id}: step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {end_time-start_time:.1f}s")

                ## aggregate the loss across all GPUs
                lossf = aggregate_value(lossf, self.cfg["general"]["device"])

                ## print and log the result only on the first GPU after aggregation
                print(f"All GPU(s): step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {end_time-start_time:.1f}s")
                if (self.gpu_id == 0 or self.gpu_id is None) and self.use_wandb:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "loss": lossf,
                            "lr": lr,
                            "token_num": self.batch_size*self.gradient_accumulation_steps*iter_num*self.cfg.model["context_window"],
                        }
                    )
        # save the final model
        if self.gpu_id == 0 or self.gpu_id is None: ## ensure only the first GPU saves the model
            self._save_model(iter_num+1) # just so it looks nicer

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()
