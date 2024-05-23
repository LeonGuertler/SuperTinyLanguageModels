"""
Contains the build functions for the embedder,
core model, lm head and the model shell.
"""

from models.core_models import GenericTransformer
from models.embedding_models import GenericEmbedder
from models.experimental.byte_level.embedding_model import ByteLevelEmbedder
from models.experimental.byte_level.model_heads import ByteLevelDecoder
from models.model_heads import AutoregressiveLMHead
from models.model_shell import ModelShell


def build_model(model_cfg=None, checkpoint=None):
    """
    Either initialize or load a model, depending on
    whether a config or checkpoint was provided
    (respectively).
    Args:
        model_cfg: model_configuration
        model_checkpoint: model_checkpoint_dict
        dataset_name: the dataset for the tokenizer
    Returns:
        model: model instance
    """

    # check if model is to be loaded
    if checkpoint is not None:
        # load model with the correct architecture
        print(f"loading model with config{checkpoint['config']}")
        model = initialize_model(checkpoint["config"]["model"])


        # load the model weights
        model.load_state_dict(checkpoint["model"])
        model.eval()

    else:
        # initialize model
        model = initialize_model(model_cfg)
        model.train()

    return model


EMBEDDING_MODEL_DICT = {"generic": GenericEmbedder, "byte_level": ByteLevelEmbedder}


def build_embedding_model(model_cfg):
    """
    Given the embedding model config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        embedding_model: embedding_model_instance
    """
    return EMBEDDING_MODEL_DICT[model_cfg["embedder"]["embedding_model_type"]](
        model_cfg=model_cfg
    )


CORE_MODEL_DICT = {"generic": GenericTransformer}


def build_core_model(model_cfg):
    """
    Given the core model config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        core_model: core_model_instance
    """
    return CORE_MODEL_DICT[model_cfg["core_model"]["core_model_type"]](
        model_cfg=model_cfg
    )


MODEL_HEAD_DICT = {"generic": AutoregressiveLMHead, "byte_level": ByteLevelDecoder}


def build_model_head(model_cfg):
    """
    Given the lm head config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        model_head: model_head_instance
    """
    return MODEL_HEAD_DICT[model_cfg["lm_head"]["lm_head_type"]](model_cfg=model_cfg)


MODEL_SHELL_DICT = {"standard": ModelShell}


def build_model_shell(model_cfg, embedding_model, core_model, model_head):
    """
    Given the model shell config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        model_shell: model_shell_instance
    """
    return MODEL_SHELL_DICT[model_cfg["model_shell_type"]](
        embedding_model=embedding_model, core_model=core_model, model_head=model_head
    )


def initialize_model(model_cfg):
    """
    Initialize the model given the configuration.
    Args:
        model_cfg: model_cfg
    Returns:
        model: model_instance
    """
    # build the embedding model
    embedding_model = build_embedding_model(model_cfg=model_cfg)

    # build the core model
    core_model = build_core_model(model_cfg=model_cfg)

    # build the model head
    model_head = build_model_head(model_cfg=model_cfg)

    # check if embedding model weights are to be shared with the model head
    if model_cfg["embedding_weight_tying"]:
        # share the weights between the token embeddings and the final
        # logit layer, following: https://paperswithcode.com/method/weight-tying
        embedding_model.token_embedder.weight = model_head.linear.weight

    # build the model shell
    model = build_model_shell(
        model_cfg=model_cfg,
        embedding_model=embedding_model,
        core_model=core_model,
        model_head=model_head,
    )

    return model
