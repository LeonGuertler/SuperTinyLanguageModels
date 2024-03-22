import os, time, math, pickle, hydra
from omegaconf import DictConfig, OmegaConf
from contextlib import nullcontext
from datasets import load_dataset
import numpy as np
import torch

from models.build_models import build_model


def dpo(x, y_w, y_l, model, ref_model, beta=1e-2):
    # x: input tensor
    # y_w: preferred completion
    # y_l: dispreferred completion
    # returns: the DPO loss

    # compute the logits for the preferred and dispreferred completions
    with torch.no_grad():
        ref_log_probs_w = ref_model(x, y_w)
        ref_log_probs_l = ref_model(x, y_l)
        ref_log_ratio = ref_log_probs_w - ref_log_probs_l
    log_probs_w = model(x, y_w)
    log_probs_l = model(x, y_l)
    log_ratio = log_probs_w - log_probs_l
    preloss = beta * (log_ratio - ref_log_ratio)
    return -torch.log(torch.sigmoid(preloss)).mean()


@torch.no_grad()
def estimate_loss(model, ref_model, eval_iters, ctx):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y_W, Y_L = model.get_batch(split)
            with ctx:
                loss = dpo(X, Y_W, Y_L, model, ref_model)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

def build_dataset(config):
    return load_dataset(config['finetuning']['dataset'], split='train')

def get_batch(dataset, tokenizer, batch_size, context_window):
    idx = np.random.randint(len(dataset), size=batch_size)
    prompts = [f"SYSTEM_TEXT:{dataset["system"][i]}\nINSTRUCTIONS:{dataset["prompt"][i]}" for i in idx]
    xs = tokenizer.encode_batch(prompts)
    y_w = [f"{dataset["chosen"][i]}" for i in idx]
    y_l = [f"{dataset["other"][i]}" for i in idx]
    return xs, y_w, y_l

def main(model_cfg: DictConfig) -> None:
    # Load the general config file
    general_cfg_path = hydra.utils.to_absolute_path("configs/general_config.yaml")
    general_cfg = OmegaConf.load(general_cfg_path)

    # Merge the general configuration with the nanoGPT configuration
    cfg = OmegaConf.merge(general_cfg, model_cfg)

    # set the random seed
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # load the dataset
    dataset = build_dataset(cfg)

    # specify the device and dtype for training
    device = "cuda"

    # build the model
    model = build_model(config=cfg, device=device)
    ref_model = build_model(config=cfg, device=device)

    # set the model in training mode
    model.train()

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        weight_decay=cfg.training.weight_decay,
    )

    # learning rate schedule
    warmup_iters = cfg.training.warmup_iters
    lr_decay_iters = cfg.training.lr_decay_iters
    min_lr = cfg.training.min_lr

    # loss function
    loss_fn = dpo

    # context for mixed precision training
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32' or 'bfloat16' or 'float16'
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    iter_num = 0
    best_val_loss = 1e9

    if cfg.logging.wandb_log:
        run_name = f"{cfg['arch']['model']}_{cfg['finetuning']['dataset']}_{cfg['finetuning']['tokenizer']}"
        import wandb

        wandb.init(
            project=cfg.logging.wandb_project,
            config=OmegaConf.to_container(cfg),
            name=run_name,
        )

    t0 = time.time()

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(
            iter_num, warmup_iters, lr_decay_iters, cfg.training.learning_rate, min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if not iter_num % cfg.logging.eval_interval:
            # evaluate the model
            losses = estimate_loss(model, ref_model, cfg.logging.eval_iters, ctx)
            print(f"Step {iter_num} Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}")

            if cfg.logging.wandb_log:
                wandb.log(
                    {
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "learning_rate": lr,
                        "step": iter_num,
                    }
                )
        for _ in range(cfg.training.gradient_accumulation_steps):
            X, Y_W, Y_L = get_batch((dataset, model.tokenizer, cfg.training.batch_size, cfg.arch.context_window))
            with ctx:
                dpo(X, Y_W, Y_L, model, ref_model, cfg.training.beta)
                loss = loss / cfg.training.gradient_accumulation_steps
            
            scaler.scale(loss).backward()

        if cfg.training.optimizer.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.optimizer.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush gradients asap
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0 
        t0 = t1
        if not iter_num % cfg.training.log_interval:
            lossf = loss.item() * cfg.training.gradient_accumulation_steps
            print(f"step {iter_num}: loss {lossf:.4f}, lr {lr:.1e}, dt {dt:.1f}s")

        iter_num += 1


        if iter_num > cfg.training.max_iters:
            break        


    # save the model
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': cfg,
    }
    print(f"saving checkpoint to {general_cfg.output_dir}")
    torch.save(checkpoint, 'ckpt.pt')



if __name__ == "__main__":
    main()
