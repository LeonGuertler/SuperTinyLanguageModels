"""
Contains the build functions for the embedder,
core model, lm head and the model shell.
"""
import torch

from models.core_models import GenericFFNSharedTransfomer, GenericTransformer
from models.embedding_models import GenericEmbedder
from models.experimental.byte_level.embedding_model import ByteLevelEmbedder
from models.experimental.byte_level.model_heads import ByteLevelDecoder
from models.experimental.byte_level.byte_model_shell import ByteModelShell
from models.experimental.hugging_face import HFEmbedder, HFLMHead, HFTransformerCore
from models.experimental.next_thought.embedding_models import HierarchicalEncoder
from models.experimental.next_thought.model_heads import VariableLengthLatentDecoder
from models.experimental.next_thought.core_models import BaselineCoreModel, Conv1dCoreModel
from models.model_heads import AutoregressiveLMHead
from models.model_shell import ModelShell
from models.core_models import GenericCProjSharedTransfomer
from models.core_models import GenericCProjFFNSharedTransfomer

from models.experimental.weight_sharing import (
    SharedInteriorFFNLoraAndCProj,
    SharedInteriorFFNLora,
)

from models.experimental.moe_weight_sharing import (
    SharedMoE
)


def build_model(model_cfg=None, checkpoint_path=None, device="cuda"):
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
        model = initialize_model(checkpoint["config"]["model"])

        # load the model weights
        model.load_state_dict(checkpoint["model"])
        current_iter = checkpoint["iter_num"]

    else:
        # initialize model
        model = initialize_model(model_cfg)
        current_iter = 0

    return model, current_iter


EMBEDDING_MODEL_DICT = {
    "generic": GenericEmbedder, 
    "byte_level": ByteLevelEmbedder,
    "hf_embedder": HFEmbedder,
    "hierarchical": HierarchicalEncoder,
    }


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


CORE_MODEL_DICT = {
    "generic": GenericTransformer,
    "generic_ffn_sharing": GenericFFNSharedTransfomer,
    "hf_core": HFTransformerCore,
    "next_thought_baseline": BaselineCoreModel,
    "conv": Conv1dCoreModel,
    "generic_cproj_shared": GenericCProjSharedTransfomer,
    "generic_ffn_qproj_sharing": GenericCProjFFNSharedTransfomer,
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
    return CORE_MODEL_DICT[model_cfg["core_model"]["core_model_type"]](
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
    return MODEL_HEAD_DICT[model_cfg["lm_head"]["lm_head_type"]](
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
    model_head = build_model_head(
        model_cfg=model_cfg,
        embedding_model=embedding_model
    )

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
