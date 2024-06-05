"""
The main generate code
"""

import hydra
import torch

from models.build_models import build_model
from models.generator import StandardGenerator


@hydra.main(config_path="configs", config_name="generate")
def main(cfg):
    """run the main eval loop"""

    # set the checkpoint path to absolute path
    cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])

    # load checkpoint from the path
    model = build_model(checkpoint=torch.load(cfg["model_ckpt"]))

    generator = StandardGenerator(model=model, generate_cfg=cfg["generator"])

    while True:
        # generate the text
        generated_text = generator.default_generate(
            input_text=input("Enter the input text: ")
        )
        print(generated_text)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
