"""Trainer class for training models with Next Token Prediction"""

import time
import wandb
from omegaconf import OmegaConf
from contextlib import nullcontext

# local imports
from trainers import utils, data_utils
from trainers.evaluator import (
    train_eval_mcq, 
    train_eval_text_modeling,
    train_eval_text_generation
)
from trainers.utils import aggregate_value, print_evaluation_results
from models.utils import print_model_stats

import numpy as np
from itertools import islice
import torch
from torch.nn.parallel import DistributedDataParallel as DDP 

import time 


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
            self.DDP_model = DDP(self.model, device_ids=[gpu_id], find_unused_parameters=True)
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


        # print training statistics
        train_token_count = f"{len(train_dataloader.dataset)/1e9:.2f}B"
        val_token_count = f"{len(val_dataloader.dataset)/1e9:.2f}B"

        print(f"Training the model on {self.cfg.model.get('dataset', None)} with {train_token_count} tokens.")


        if self.use_wandb and (self.gpu_id == 0 or not self.dist): ## ensures that only the first GPU logs to wandb
            self._setup_logging(
                total_parameter_count_str=total_params_formated,
                train_token_count=train_token_count,
                val_token_count=val_token_count
            )


    def _setup_logging(
        self, 
        total_parameter_count_str=None,
        train_token_count=None,
        val_token_count=None
    ):
        # check if run_name was provided
        if self.cfg["general"]["logging"].get("run_name", None) is not None:
            run_name = self.cfg["general"]["logging"]["run_name"] + \
                f" (Size: {total_parameter_count_str})"
        else:
            # provide a generic (hopefully descriptive) run name if none was provided
            run_name = (
                f"Unname_Model_{self.cfg.trainer['dataset']}"
                f"_{self.cfg.model['vocab_size']}"
                f"_Parameters_{total_parameter_count_str}"
                f"_TrainTokens_{train_token_count}"
            )


        # Specific the tags
        tags = [
            f"Core-{self.cfg.model.get('core_model_type', None)}",
            f"Shell-{self.cfg.model.get('model_shell_type', None)}",
            f"Embebdding-{self.cfg.model.get('embedding_model_type', None)}",
            f"LM_Head-{self.cfg.model.get('lm_head_type', None)}",
            f"Dataset-{self.cfg.trainer.get('dataset', None)}",
            f"Vocab_size-{self.cfg.model.get('vocab_size', None)}",
            f"Parameters-{total_parameter_count_str.split('.')[0]}",
            f"TrainTokens-{train_token_count}",
            f"ValTokens-{val_token_count}"
        ]


        wandb.init(
            project=self.cfg["general"]["logging"].get("wandb_project", "SuperTinyLanguageModels"), 
            config=OmegaConf.to_container(self.cfg),
            name=run_name,
            tags=tags,
            group=self.cfg["general"]["logging"].get("group_name", "General")
        )
        print("Weight and Biases Initialized")

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
    def estimate_performance(self, iter_num, eval_iters=None):
        """Estimate the loss"""
        # Initialize eval results
        eval_results = {
            "iter": iter_num,
            "token_num": (
                self.batch_size
                * self.gradient_accumulation_steps
                * iter_num
                * self.cfg.model["context_window"]
            ),
        }

        # Make sure the model is in eval mode
        self.model.eval()

        # Initialize accumulators
        total_loss = 0.0
        total_bytes = 0
        total_token_count = 0  # For regular loss and perplexity

        # Initialize accumulators for byte-level metrics
        total_byte_loss = 0.0

        # Determine the device
        device = self.gpu_id if self.gpu_id is not None else self.model.device

        for i, (x, y) in enumerate(self.val_dataloader):
            # Move tensors to the appropriate device
            x = x.to(device)
            y = y.to(device)

            with self.ctx:
                output, _, _ = self.model(x)
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    y.view(-1),
                    reduction='sum'
                )

            # Accumulate token-level metrics
            total_loss += loss.item()
            total_token_count += y.numel()  # Use numel() for total tokens

            if self.evaluate_byte_metrics:
                # Compute byte counts for current batch
                y_tokens = y.view(-1).cpu().numpy()
                byte_counts = self.model.embedding_model.get_byte_lengths(y_tokens)
                batch_byte_count = byte_counts.sum()
                total_bytes += batch_byte_count

                # Accumulate byte-level loss
                avg_loss_per_token = loss.item() / y.numel()
                total_byte_loss += avg_loss_per_token * batch_byte_count

            # Check if we've reached the evaluation iteration limit
            if eval_iters is not None and i >= eval_iters:
                break

        # Calculate average metrics
        avg_loss = total_loss / total_token_count if total_token_count > 0 else float('inf')
        avg_perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')  # Avoid overflow

        # Store in eval_results
        eval_results["Validation/Loss"] = aggregate_value(avg_loss, self.cfg.general.device)
        eval_results["Validation/Perplexity"] = aggregate_value(avg_perplexity, self.cfg.general.device)

        if self.evaluate_byte_metrics:
            if total_bytes > 0:
                avg_byte_loss = aggregate_value(total_byte_loss, self.cfg.general.device) / total_bytes
                avg_byte_perplexity = np.exp(avg_byte_loss) if avg_byte_loss < 100 else float('inf')  # Avoid overflow
            else:
                avg_byte_loss = float('inf')
                avg_byte_perplexity = float('inf')

            # Byte-normalized metrics
            eval_results["Validation/Loss (Bytes)"] = avg_byte_loss
            eval_results["Validation/Perplexity (Bytes)"] = avg_byte_perplexity

        # get the mcq eval results
        eval_results.update(
            train_eval_mcq(
                model=self.model,
                num_samples=self.cfg["trainer"]["eval"].get("mcq_num_samples", None),
                benchmark_list=self.cfg["trainer"]["eval"].get("mcq_benchmarks", []),
            )
        )
        # get the text modeling eval results
        if self.cfg["trainer"]["eval"].get("text_modeling_eval", False):
            eval_results.update(
                train_eval_text_modeling(
                    model=self.model,
                    topic_list=self.cfg["trainer"]["eval"].get("text_modeling_topics", []),
                )
            )

        if self.cfg["trainer"]["eval"].get("text_generation_eval", False):
            text_generation_results, text_generation_sample_html = train_eval_text_generation(
                model=self.model
            )
            eval_results.update(text_generation_results)

            # log the generated text
            if (self.gpu_id==0 or self.gpu_id is None) and self.use_wandb:
                wandb.log(
                    {
                        "Generated Text": wandb.Html(
                            text_generation_sample_html
                        )
                    },
                    step=eval_results["token_num"]
                )

        # set model back into train mode
        self.model.train()

        # print the eval results
        print_evaluation_results(
            iter_num=iter_num,
            eval_results=eval_results
        )

        # log the evaluation results
        if (self.gpu_id==0 or self.gpu_id is None) and self.use_wandb: # ensure only the first GPU logs
            wandb.log(
                eval_results,
                step=eval_results["token_num"]
            )

        return eval_results


    def _run_step(self):
        """Run a single step of training with gradient accumulation."""
        self.optimizer.zero_grad()  # Clear gradients at the start of accumulation

        accumulated_loss = 0
        additional_info_dict = {}
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
                    output, loss, additional_info_sub_dict = self.DDP_model(x)
                    # loss = self.loss_fn(output, y)
                    # if aux_loss is not None:
                    #     loss += aux_loss


                # Scale loss to simulate larger effective batch size
                loss = loss / self.gradient_accumulation_steps
                for key in additional_info_sub_dict:
                    additional_info_dict[key] = additional_info_dict.get(key, 0) + additional_info_sub_dict[key]/self.gradient_accumulation_steps
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

        return accumulated_loss, additional_info_dict

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
                not iter_num % self.cfg["trainer"]["eval_interval"] \
                and self.cfg["trainer"].get("run_eval", True)
            ): # run on first iter to prevent bugs causing it to crash
                self.estimate_performance(
                    iter_num=iter_num,
                    eval_iters=self.cfg["trainer"].get("eval_iters", 100)
                )

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


            lossf, additional_info_dict = self._run_step() 
            end_time = time.time()
            if not iter_num % self.cfg["trainer"]["log_interval"] and iter_num > 0:
                ## uncomment the following line to print the loss on all GPUs
                # print(f"GPU {self.gpu_id}: step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {end_time-start_time:.1f}s")

                ## aggregate the loss across all GPUs
                lossf = aggregate_value(lossf, self.cfg["general"]["device"])

                ## print and log the result only on the first GPU after aggregation
                print(f"All GPU(s): step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {end_time-start_time:.1f}s")
                if (self.gpu_id == 0 or self.gpu_id is None) and self.use_wandb:
                    token_num = self.batch_size*self.gradient_accumulation_steps*iter_num*self.cfg["model"]["context_window"]
                    wandb.log(
                        {
                            "iter": iter_num,
                            "loss": lossf,
                            "lr": lr,
                            "token_num": token_num,
                        },
                        step=token_num
                    )

                    # log all additional information
                    if len(additional_info_dict) > 0:
                        # refactor keys for better logging structure
                        wandb.log(
                            {
                                f"additional_info/{key}": additional_info_dict[key] \
                                for key in additional_info_dict.keys()
                            },#.update({"token_num": token_num})
                            step=token_num
                        )

        # save the final model
        if self.gpu_id == 0 or self.gpu_id is None: ## ensure only the first GPU saves the model
            self._save_model(iter_num+1) # just so it looks nicer

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()