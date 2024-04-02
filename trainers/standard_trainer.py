"""Trainer class for training models with Next Token Prediction"""


import torch
import hydra, os
from omegaconf import DictConfig, OmegaConf

# get local imports
from models.build_models import build_model

from trainers.loss_fn import build_loss_fn
from trainers.scheduler import (
    CosineScheduler,
)
from trainers.utils import (
    set_seed
)

class BaseTrainer:
    """
    The BaseTrainer accepts as input:
        - model: the model to train
        - optimizer: the optimizer to use
        - scheduler: the learning rate scheduler
        - dataloader: the dataloader to use
        - loss_fn: the loss function to use
    and trains the model.
    """
    def __init__(
        self,
        cfg,
        model,
        optimizer,
        scheduler,
        dataloader,
        loss_fn
    ):
        """
        TODO
        """
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.loss_fn = loss_fn


        # For training, always force the device to be cuda
        assert torch.cuda.is_available(), "CUDA must be available for training"
        self.device = torch.device("cuda")

        # define dtype
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        # set scaler if applicable
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))


        # push model to device
        self.model.to(self.device)


        self.use_wandb = self.cfg["general"]["logging"]["wandb_log"]
        if self.use_wandb:
            import wandb
            # set run name
            run_name = f"{self.cfg['model']['model']}_{self.cfg['trainer']['dataset']}_{self.cfg['model']['tokenizer']}"
            wandb.init(
                project=self.cfg["general"]["logging"]["wandb_project"], 
                config=OmegaConf.to_container(self.cfg),
                name=run_name
            )
            wandb.init(project=self.cfg["general"]["logging"]["wandb_project"])

        # set seed
        set_seed(self.cfg["general"]["seed"])

        # get original path from hydra
        try:
            self.original_cwd = hydra.utils.get_original_cwd()
        except:
            self.original_cwd = os.getcwd()
        # check if the data has been preprocessed
        # TODO 
        self.is_processed = False

    def preprocess_data(self):
        """
        Preprocess the data
        """
        print("Preprocessing the training data")
        self.dataloader.prepare_data(
            tokenizer=self.model.tokenizer
        )
        self.is_processed = True 
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        eval_iters = self.cfg["training"]["eval_iters"]

        # set model to eval mode
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.dataloader.get_batch(split)
                with self.ctx:
                    logits = self.model(X)
                    loss = self.loss_fn(logits, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        # set model back to train mode
        self.model.train()
        return out


    def train(self):

        # check if the data is processed
        assert self.is_processed, "Data must be processed before training. Please call Trainer.preprocess_data()"
        iter_num = 0
        t0 = time.time()

        while True:
            # determine the learning rate for this iteration
            lr = self.scheduler.get_lr(iter_num)

            # apply the learning rate to the optimizer
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loass on train/val sets
            if not iter_num % self.cfg["general"]["eval_interval"]:
                losses = self.estimate_loss(
                    self.cfg["training"]["eval_iters"],
                )
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if self.use_wandb:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                    })
                    
            # save every checkpoint_interval iterations
            if not iter_num % self.cfg["training"]["checkpoint_interval"] and iter_num > 0:
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "iter_num": iter_num,
                    "config": self.cfg,
                }
                checkpoint_path = os.path.join(
                    self.original_cwd,
                    self.cfg["general"]["checkpoint_dir"],
                    f"ckpt_{iter_num}.pt"

                )
                print(f"saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)



            # actually train the model
            for micro_step in range(self.cfg["training"]["gradient_accumulation_steps"]):
                # get training batch
                X, Y = self.dataloader.get_batch("train")
                with self.ctx:
                    logits = self.model(X)
                    loss = self.loss_fn(logits, Y)
                    loss = loss / self.cfg["training"]["gradient_accumulation_steps"]

                self.scaler.scale(loss).backward()

            # clip gradients if necessary
            if self.cfg["training"]["optimizer"]["grad_clip"] != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg["training"]["optimizer"]["grad_clip"]
                )

            # step the optimizer and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # flush gradients
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0 
            t0 = t1
            if not iter_num % self.cfg["training"]["log_interval"]:
                lossf = loss.item() * self.cfg["training"]["gradient_accumulation_steps"]
                print(f"step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {dt:.1f}s")

            iter_num += 1


            if iter_num > self.cfg["training"]["max_iter"]:
                break

        # save the final model
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "config": self.cfg,
        }
        checkpoint_path = os.path.join(
            self.original_cwd,
            self.cfg["general"]["checkpoint_dir"],
            f"ckpt.pt"

        )
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)




