import os, time, math, pickle, hydra, torch, tiktoken
from omegaconf import DictConfig, OmegaConf
from contextlib import nullcontext

from models.build_models import build_model

from contextlib import nullcontext




@hydra.main(config_path="configs/test/", config_name="baseline.yaml")
def main(cfg: DictConfig) -> None:
    model = build_model(
        ckpt_path=cfg["model_path"]
    )
    model.eval()
    
    # generation hyperparameters
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    device = "cpu"
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    with torch.no_grad():
        with ctx:
            print(
                model.generate(
                    "The quick brown fox jumps over the lazy dog",
                    cfg["max_new_tokens"],
                    cfg["temperature"],
                    cfg['top_k']
                )
            )


if __name__ == "__main__":
    main()