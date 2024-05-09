"""
Contains the functions to build the actual model.
"""

from models.core_models import (
    GenericTransformer,
    ModernFFNSharingTransformer,
    ModernTransformer,
    StandardTransformer,
)
from models.shells import build_shell


def build_model(cfg=None, checkpoint=None):
    """
    Either initialize or load a model, depending on
    whether a config or checkpoint was provided
    (respectively).
    Args:
        cfg: model_configuration
        model_checkpoint: model_checkpoint_dict
    Returns:
        model: model instance
    """

    # check if model is to be loaded
    if checkpoint is not None:
        # load model with the correct architecture
        model = initialize_model(checkpoint["config"])

        # load the model weights
        model.load_state_dict(checkpoint["model"])
        model.eval()

    else:
        # initialize model
        model = initialize_model(cfg)
        model.train()

    return model


CORE_MODEL_DICT = {
    "baseline": StandardTransformer,
    "modern": ModernTransformer,
    "modern_ffn_sharing": ModernFFNSharingTransformer,
    "generic": GenericTransformer,
}


def build_core_model(cfg):
    """
    Given the core model config, build it.
    Args:
        cfg: cfg
    Returns:
        core_model: core_model_instance
    """
    return CORE_MODEL_DICT[cfg["core_model"]["core_model_type"]](cfg=cfg)


def initialize_model(cfg):
    """
    Initializes the model from the configuration
    Args:
        cfg: main config
    Returns:
        model: model instance
    """
    # build the core model
    core_model = build_core_model(cfg)

    # build the model shell
    model = build_shell(cfg, core_model)

    return model
