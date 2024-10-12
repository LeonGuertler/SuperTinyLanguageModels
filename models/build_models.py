"""
Contains the build functions for the embedder,
core model, lm head and the model shell.
"""
import torch
from omegaconf import OmegaConf


from models.core_models import GenericTransformer
from models.embedding_models import GenericEmbedder
from models.experimental.byte_level.embedding_model import ByteLevelEmbedder
from models.experimental.byte_level.model_heads import ByteLevelDecoder
from models.experimental.byte_level.byte_model_shell import ByteModelShell
from models.experimental.hugging_face import HFEmbedder, HFLMHead, HFTransformerCore
from models.model_heads import AutoregressiveLMHead
from models.model_shell import ModelShell

from models.experimental.weight_sharing import (
    SharedInteriorFFNLoraAndCProj,
    SharedInteriorFFNLora,
)

from models.experimental.moe_weight_sharing import (
    SharedMoE
)


def build_model(model_cfg=None, checkpoint_path=None, device="cuda", **kwargs):
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
    if checkpoint_path is not None:
        # load model with the correct architecture
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(device),
        )

        # update the attention type if provided
        if checkpoint["config"]["model"].get("attention_type", None) is None:
            cfg = OmegaConf.create({"attention_type": f"{kwargs.get('attention_type', 'standard')}"})
            checkpoint['config']['model'] = OmegaConf.merge(cfg, checkpoint['config']['model'])
        elif checkpoint["config"]["model"].get("attention_type", None) == "standard":
            checkpoint["config"]["model"]["attention_type"] = f"{kwargs.get('attention_type', 'standard')}"

        model = initialize_model(checkpoint["config"]["model"])

        # load the model weights
        model.load_state_dict(checkpoint["model"])

        loaded_train_config = {
            "optimizer": checkpoint["optimizer"],
            "iter_num": checkpoint["iter_num"],
            "config": checkpoint["config"]
        }

    else:
        # initialize model
        model = initialize_model(model_cfg)
        loaded_train_config = None

    return model, loaded_train_config


EMBEDDING_MODEL_DICT = {
    "generic": GenericEmbedder, 
    "byte_level": ByteLevelEmbedder,
    "hf_embedder": HFEmbedder,
    }


def build_embedding_model(model_cfg):
    """
    Given the embedding model config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        embedding_model: embedding_model_instance
    """
    return EMBEDDING_MODEL_DICT[model_cfg["embedding_model_type"]](
        model_cfg=model_cfg
    )


CORE_MODEL_DICT = {
    "generic": GenericTransformer,
    "hf_core": HFTransformerCore,

    # experimental
    "ffn_lora_sharing": SharedInteriorFFNLora,
    "ffn_lora_sharing": SharedInteriorFFNLoraAndCProj,
    "ffn_lora_sharing_moe": SharedMoE,
}


def build_core_model(model_cfg):
    """
    Given the core model config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        core_model: core_model_instance
    """
    return CORE_MODEL_DICT[model_cfg["core_model_type"]](
        model_cfg=model_cfg
    )


MODEL_HEAD_DICT = {
    "generic": lambda model_cfg, embedding_model: AutoregressiveLMHead(model_cfg=model_cfg), 
    "byte_level": lambda model_cfg, embedding_model: ByteLevelDecoder(model_cfg=model_cfg), 
    "hf_head": lambda model_cfg, embedding_model: HFLMHead(model_cfg=model_cfg),
    "latent_2_seq": lambda model_cfg, embedding_model: VariableLengthLatentDecoder(
        model_cfg=model_cfg,
        embedding_model=embedding_model
    ), 
    }


def build_model_head(model_cfg, embedding_model=None):
    """
    Given the lm head config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        model_head: model_head_instance
    """
    return MODEL_HEAD_DICT[model_cfg["lm_head_type"]](
        model_cfg=model_cfg, 
        embedding_model=embedding_model
    )


MODEL_SHELL_DICT = {
    "standard": ModelShell,
    "byte_shell": ByteModelShell
}


def build_model_shell(model_cfg, embedding_model, core_model, model_head):
    """
    Given the model shell config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        model_shell: model_shell_instance
    """
    return MODEL_SHELL_DICT[model_cfg["model_shell_type"]](
        model_cfg=model_cfg,
        embedding_model=embedding_model, 
        core_model=core_model, 
        model_head=model_head,
    )


MODEL_WEIGHT_INIT_DICT = {
    "xavier": torch.nn.init.xavier_normal_,
    "kaiming": torch.nn.init.kaiming_normal_,
    "none": None,
}
def build_model_initialization_function(model_cfg):
    """
    Given the model initialiation config, build it.
    """
    return MODEL_WEIGHT_INIT_DICT[
        model_cfg.get("initialization_fn", "kaiming")
    ]


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
    model_head = build_model_head(
        model_cfg=model_cfg,
        embedding_model=embedding_model
    )


    # build the model shell
    model = build_model_shell(
        model_cfg=model_cfg,
        embedding_model=embedding_model,
        core_model=core_model,
        model_head=model_head,
    )

    # initialize the model weights
    init_fn_type = model_cfg.get("initialization_fn", "kaiming")
    if init_fn_type != None:
        init_fn = build_model_initialization_function(model_cfg)

        def init_weights(m):
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
                init_fn(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)


    return model
