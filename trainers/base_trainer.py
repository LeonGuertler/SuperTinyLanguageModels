"""Trainer class for training models with Next Token Prediction"""

import time

import torch
import wandb
from omegaconf import OmegaConf

from trainers import utils

torch.multiprocessing.set_start_method('spawn')
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
        dataloader,
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
        self.gradient_accumulation_steps = (
            cfg["training"]["gradient_accumulation_steps"]
        )
        self.scaler = None
        self.use_wandb = cfg["general"]["logging"]["wandb_log"]
        self.checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]

        # For training, always force the device to be cuda
        assert torch.cuda.is_available(), "CUDA must be available for training"
        self.ctx = self._setup_ctx()
        if self.use_wandb:
            self._setup_logging()

    def _setup_logging(self):
        # set run name
        run_name = (
            f"{self.cfg.model.embedding_model_type}_{self.cfg.model.core_model_type}_{self.cfg.model.lm_head_type}"+
            f"_{self.cfg.training.dataset}_{self.cfg.model.tokenizer}"+
            f"_{self.cfg.model.vocab_size}"
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
        self.dataloader.prepare_data(
            embedding_model=self.model.embedding_model
        )

    @torch.no_grad()
    def estimate_performance(self, model, tokenizer, eval_iters=1000):
        """Estimate the loss"""
        loss = {}
        perplexity = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            perplexities = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = self.dataloader.get_batch(split)
                decoded_xs = [tokenizer.decode(x[i].tolist()) for i in range(x.size(0))]
                token_lengths = [x.size(1) for _ in range(x.size(0))]
                char_lengths = [len(decoded_x) for decoded_x in decoded_xs]
                with self.ctx:
                    output, _ = model(x)
                    losses[i] = self.loss_fn(output, y)
                    perplexities[i] = torch.exp(
                        losses[i] * sum(token_lengths) / sum(char_lengths)
                    )
            loss[split] = losses.mean().item()
            perplexity[split] = perplexities.mean().item()
        model.train()
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
        grad_clip = self.cfg.training.optimizer.grad_clip
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
        torch.save(checkpoint, checkpoint_path)

    def run_training_loop(self):
        """Run the training loop"""
        for iter_num in range(self.cfg.training.max_iter):
            start_time = time.time()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(self.optimizer, iter_num)
            else:
                lr = self.optimizer.param_groups[0]["lr"]
            dropout = self.dropout_scheduler.step(self.model, iter_num)
            # estimate the loss on the train/val sets
            if not (iter_num+1) % self.cfg.training.eval_interval:
                losses, perplexities = self.estimate_performance(
                    self.model, tokenizer=self.model.embedding_model.tokenizer
                )
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f},"
                    f" val loss {losses['val']:.4f}"
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

            loss = self._run_step()
            end_time = time.time()
            if not (iter_num+1)\ % self.cfg.training.log_interval:
                lossf = (
                    loss.item() * self.gradient_accumulation_steps
                )  # TODO double check
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
