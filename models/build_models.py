"""
A simplistic model builder for building models 
from scratch or from checkpoints
"""

import torch
from models.architectures.baseline import BaseGPT
from models.architectures.mod_baseline import ModernGPT
from models.architectures.profiler import ProfilerGPT
from models.architectures.seq2seq_transformer import Seq2SeqModel
from models.architectures.shared_ffn import SharedFFN
from models.architectures.shared_ffn_lora import SharedFNNLora

MODEL_CLASSES = {
    "baseline": BaseGPT,
    "profiler": ProfilerGPT,
    "sharedfnn": SharedFFN,
    "sharedfnnlora": SharedFNNLora,
    "seq2seq_baseline": Seq2SeqModel,
    "modern_baseline": ModernGPT,
}


def build_model(cfg=None, model_checkpoint=None):
    """
    Build model from scratch or load model from checkpoint
    Args:
        cfg: model configuration
        model_checkpoint: model weight dict
    Returns:
        model: model instance
    """
    # check if model is loaded
    if model_checkpoint is not None:
        # load model with correct architecture
        model = MODEL_CLASSES[model_checkpoint["config"]["model"]["model"]](
            cfg=model_checkpoint["config"]["model"]
        )

        # check for device, here cpu is ok
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        # load model weights
        model.load_state_dict(model_checkpoint["model"])  # , device_map=device_name)
        model.eval()

        return model

    else:
        # build model from scratch
        model = MODEL_CLASSES[cfg["model"]](cfg=cfg)

        return model
