"""
The main generate code
"""

import hydra
import torch

from models.build_models import build_model
from models.generator import build_generator


@hydra.main(config_path="configs", config_name="generate")
def main(cfg):
    """run the main eval loop"""

    # set the checkpoint path to absolute path
    cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])

    # load checkpoint from the path
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    model, _ = build_model(
        checkpoint_path=cfg["model_ckpt"],
        device=device
    )
                        
    # put model into eval mode
    model.eval()

    generator = build_generator(
        model=model,
        generate_cfg=cfg["generator"],
        device=device
    )

    # generate the text
    for _ in range(5): # generate 5 samples
        generated_text = generator.default_generate(
            input_text=cfg["generator"]["input_text"]
        )
        print("".join(generated_text))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
