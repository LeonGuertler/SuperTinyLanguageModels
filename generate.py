"""
The main generate code
"""
import hydra
import torch 


from models.generator import StandardGenerator
from models.build_models import build_model


@hydra.main(config_path="configs/generate", config_name="baseline")
def main(cfg):
    """ run the main eval loop """

    # set the checkpoint path to absolute path
    cfg["ckpt_path"] = hydra.utils.to_absolute_path(
        cfg["ckpt_path"]
    )

    # load checkpoint from the path
    model = build_model(
        model_checkpoint=torch.load(cfg["ckpt_path"])
    )

    generator = StandardGenerator(
        model=model,
        generate_cfg=cfg["generate_cfg"]
    )

    # generate the text
    generated_text = generator.default_generate(
        input_text=cfg["input_text"]
    )


    print(generated_text)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter


    #python3 generate.py ckpt_path=outputs/2024-04-11/11-34-28/checkpoints/ckpt_9999.pt
