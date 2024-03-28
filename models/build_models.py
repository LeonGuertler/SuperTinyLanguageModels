import torch
from models.baseline import baseGPT
from models.the_10m_model.the_10m_model import the10mmodel
from models.baseline_ffn_sharing import baseGPT as baseGPT_ffn_sharing
from models.baseline_ffn_sharing_lora import baseGPT as baseGPT_ffn_sharing_lora
import hydra.utils


MODEL_CLASSES = {
    "baseline": baseGPT,
    "baseline_ffn_sharing": baseGPT_ffn_sharing,
    "baseline_ffn_sharing_lora": baseGPT_ffn_sharing_lora,
    "the_10m_model": the10mmodel,
}


def build_model(config=None, ckpt_path=None):
    # check if model is loaded
    if ckpt_path is not None:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")  # config["ckpt_path"],

        # load model with correct architecture
        model = MODEL_CLASSES[checkpoint["config"]["arch"]["model"]](
            config=checkpoint["config"]
        )

        # load model weights
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    else:
        # build model from scratch
        model = MODEL_CLASSES[config["arch"]["model"]](config=config)

        return model
