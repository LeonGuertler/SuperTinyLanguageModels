"""
Contains the functions to build the actual model.
"""

from models.core_models import (
    StandardTransformer
)

from models.autoregressive_model_shell import (
    AutoregressiveModelShell
)


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
        model.load_state_dict(checkpoint['model'])
        model.eval()
    
    else:
        # initialize model
        model = initialize_model(cfg)
        model.train()

    return model 



CORE_MODEL_DICT = {
    "baseline": StandardTransformer
}
def build_core_model(model_cfg):
    """
    Given the core model config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        core_model: core_model_instance
    """
    return CORE_MODEL_DICT[model_cfg['core_model']['core_model_type']](
        model_cfg=model_cfg
    )


MODEL_SHELL_DICT = {
    "autoregressive": AutoregressiveModelShell
}

def build_shell(model_cfg, core_model):
    """
    Given the model shell config, build it.
    Args:
        model_cfg: model_cfg
        tokenizer: tokenizer_instance
        core_model: core_model_instance
    Returns:
        model_shell: model_shell_instance
    """
    return MODEL_SHELL_DICT[model_cfg["model_shell"]['shell_type']](
        model_cfg=model_cfg,
        core_model=core_model,
    )


def initialize_model(cfg):
    """
    Initializes the model from the configuration
    Args:
        cfg: main config 
    Returns:
        model: model instance
    """
    # build the core model
    core_model = build_core_model(cfg['model'])


    # build the model shell
    model = build_shell(
        model_cfg=cfg['model'],
        core_model=core_model,
    )

    return model 

