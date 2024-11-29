from trainers import base_trainer
import wandb
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from trainers.utils import print_evaluation_results, aggregate_value

NUM_GRADIENTS = 10
FIXED_ESTIMATE_FREQ = 100
class GradiantVarianceTrainer(base_trainer.BaseTrainer):
    """A variant of the base trainer that attempts to measure the variance of gradient batches
    
    Note that this takes a lot of memory and should be used to find optimal points for
    using early dropout"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grads = [] # Running list of full model gradients

    def _run_step(self, fixed_estimate=False):
        """Run a single step of training with gradient accumulation."""
        self.optimizer.zero_grad()  # Clear gradients at the start of accumulation

        accumulated_loss = 0
        for i in range(self.gradient_accumulation_steps):
            # Get the next batch
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

        # Once gradients are accumulated, step 
        if self.cfg.trainer.optimizer.grad_clip > 0:
            # Unscale the gradients of the optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Clip the gradients with normalization
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.optimizer.grad_clip)
        
        if not fixed_estimate:
            # Perform a single optimization step
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # Collect the gradients for each layer
        step_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                step_grads.append(param.grad.detach().flatten().clone())
        self.grads.append(step_grads)
        self.grads = self.grads[-NUM_GRADIENTS:]
        self.optimizer.zero_grad()  # Reset gradients after update
        if not fixed_estimate:
            self._report_grad_var(fixed_estimate=False)
        return accumulated_loss     

    def _report_grad_var(self, fixed_estimate=False):
        """Report the variance of the gradient directions using cosine similarity between corresponding layers."""
        if len(self.grads) == NUM_GRADIENTS:
            # Normalize gradients to focus on direction
            normalized_grads = [[grad / grad.norm() for grad in step] for step in self.grads]

            # Compute pairwise cosine similarities for corresponding layers
            similarities = []
            for layer_idx in range(len(normalized_grads[0])):  # Iterate over layers
                for i in range(NUM_GRADIENTS):
                    for j in range(i + 1, NUM_GRADIENTS):
                        sim = F.cosine_similarity(
                            normalized_grads[i][layer_idx], 
                            normalized_grads[j][layer_idx], 
                            dim=0
                        )
                        similarities.append(sim.item())
            if fixed_estimate:
                wandb.log({
                    "grad_dir_var_fixed": sum(similarities) / len(similarities)
                })
            else:
                wandb.log({
                    "grad_dir_var": sum(similarities) / len(similarities)
                })


    def run_training_loop(self):
        """Run the training loop"""
        for iter_num in range(self.cfg.trainer.training.max_iters):
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(self.optimizer, iter_num)
            else:
                lr = self.optimizer.param_groups[0]["lr"]
            dropout = self.dropout_scheduler.step(self.model, iter_num)
            # estimate the loss on the train/val sets
            if (
                not iter_num % self.cfg.trainer.training.eval_interval
            ): # run on first iter to prevent bugs causing it to crash
                eval_results, benchmark_results = self.estimate_performance()

                # print the evals as table
                # evals format is d1: type d2: train/val
                print_evaluation_results(
                    iter_num=iter_num, 
                    eval_results=eval_results, 
                    benchmark_results=benchmark_results
                )

                # Log to wandb
                if (self.gpu_id == 0 or self.gpu_id is None) and self.use_wandb:  # ensure only the first GPU logs
                    log_dict = {"iter": iter_num, "lr": lr, "dropout": dropout}
                    log_dict.update(eval_results)  # Directly add evals to the log dictionary
                    log_dict.update({k:v for k,v in benchmark_results.items()}) # Add benchmark results to the log dictionary

                    wandb.log(log_dict)

            # save checkpoints
            if (
                not iter_num % self.cfg.trainer.training.checkpoint_interval
                and iter_num > 0
                and (
                    self.gpu_id == 0
                    or self.gpu_id == None
                 ) ## ensure only the first GPU prints
            ):
                self._save_model(iter_num)


            lossf = self._run_step() ## set the 'epoch' to ensure shuffle
            if not iter_num % self.cfg.trainer.training.log_interval and iter_num > 0:
                ## uncomment the following line to print the loss on all GPUs
                ## aggregate the loss across all GPUs
                lossf = aggregate_value(lossf, self.cfg.general.device)

                ## print and log the result only on the first GPU after aggregation
                if (self.gpu_id == 0 or self.gpu_id is None) and self.use_wandb:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "loss": lossf,
                            "lr": lr,
                            "dropout": dropout,
                        }
                    )
            
            if not iter_num % FIXED_ESTIMATE_FREQ:
                history = self.grads.copy()
                self._run_step(fixed_estimate=True)
                self._report_grad_var(fixed_estimate=True)
                self.grads = history # Reset the history to avoid biasing the estimate
