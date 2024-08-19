"""
Contains the build functions for the embedder,
core model, lm head and the model shell.
"""

from models import model_shell
from models.cast_configs import ModelShellConfigMap
from models.core_models import GenericFFNSharedTransfomer, GenericTransformer
from models.embedding_models import GenericEmbedder
from models.experimental.byte_level.byte_model_shell import (
    ByteModelShell,
    ByteShellConfig,
)
from models.experimental.byte_level.embedding_model import ByteLevelEmbedder
from models.experimental.byte_level.model_heads import ByteLevelDecoder
from models.experimental.hugging_face import HFEmbedder, HFLMHead, HFTransformerCore
from models.experimental.next_thought.core_models import (
    BaselineCoreModel,
    Conv1dCoreModel,
)
from models.experimental.next_thought.embedding_models import HierarchicalEncoder
from models.experimental.next_thought.model_heads import VariableLengthLatentDecoder
from models.model_heads import AutoregressiveLMHead


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
        model = initialize_model(checkpoint["config"]["model"])

        # load the model weights
        model.load_state_dict(checkpoint["model"])

    else:
        # initialize model
        model = initialize_model(model_cfg)

    return model


EMBEDDER_DICT = {
    "generic": GenericEmbedder,
    "hf_embedder": HFEmbedder,
    "nt_embedder": HierarchicalEncoder,
    "byte_embedder": ByteLevelEmbedder,
}


def build_embedding_model(
    model_cfg: ModelShellConfigMap | ByteModelShell,
) -> GenericEmbedder:
    """
    Given the embedding model config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        embedding_model: embedding_model_instance
    """
    embedder_cfg = model_cfg.embedding_model
    embedder_type = model_cfg.embedding_model.embedding_model_type
    match embedder_type:
        case "byte_embedder":
            return ByteLevelEmbedder(
                embedder_cfg=embedder_cfg,
                byte_cfg=model_cfg,
                hidden_dim=model_cfg.hidden_dim,
                vocab_size=model_cfg.vocab_size,
            )
        case "hf_embedder":
            return HFEmbedder(model_cfg=embedder_cfg)
        case "nt_embedder":
            return HierarchicalEncoder(
                embedder_cfg=embedder_cfg,
                vocab_size=model_cfg.vocab_size,
                hidden_dim=model_cfg.hidden_dim,
                context_window=model_cfg.context_window,
                positional_encoding_type=model_cfg.positional_encoding_type,
            )
        case "generic":
            return GenericEmbedder(
                embedder_cfg=embedder_cfg,
                vocab_size=model_cfg.vocab_size,
                hidden_dim=model_cfg.hidden_dim,
                context_window=model_cfg.context_window,
                positional_encoding_type=model_cfg.positional_encoding_type,
            )


CORE_MODEL_DICT = {
    "generic": GenericTransformer,
    "generic_ffn_sharing": GenericFFNSharedTransfomer,
    "hf_core": HFTransformerCore,
    "next_thought_baseline": BaselineCoreModel,
    "conv": Conv1dCoreModel,
}


def build_core_model(
    model_cfg: model_shell.ModelShellConfig | ByteModelShell,
) -> GenericTransformer:
    """
    Given the core model config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        core_model: core_model_instance
    """
    core_model_cfg = model_cfg.core_model
    core_model_type = core_model_cfg.core_model_type
    match core_model_type:
        case "generic":
            return GenericTransformer(
                hidden_dim=model_cfg.hidden_dim,
                context_window=model_cfg.context_window,
                core_model_cfg=core_model_cfg,
            )
        case "generic_ffn_sharing":
            return GenericFFNSharedTransfomer(
                hidden_dim=model_cfg.hidden_dim,
                context_window=model_cfg.context_window,
                core_model_cfg=core_model_cfg,
            )
        case "hf_core":
            return HFTransformerCore(model_cfg=core_model_cfg)
        case "next_thought_baseline":
            return BaselineCoreModel(model_cfg=core_model_cfg)
        case "conv":
            return Conv1dCoreModel()


MODEL_HEAD_DICT = {
    "hf_lm_head": HFLMHead,
    "nt_lm_head": VariableLengthLatentDecoder,
    "byte_lm_head": ByteLevelDecoder,
    "generic": AutoregressiveLMHead,
}


def build_model_head(model_cfg: ModelShellConfigMap, embedding_model: GenericEmbedder):
    """
    Given the model head config, build it.
    Args:
        model_cfg: model_cfg
        embedding_model: embedding_model_instance
    Returns:
        model_head: model_head_instance
    """
    model_head_cfg = model_cfg.model_head
    model_head_type = model_head_cfg.model_head_type
    match model_head_type:
        case "hf_lm_head":
            return HFLMHead(model_cfg=model_head_cfg)
        case "nt_lm_head":
            return VariableLengthLatentDecoder(
                model_cfg=model_head_cfg, embedding_model=embedding_model
            )
        case "generic":
            return AutoregressiveLMHead(
                hidden_dim=model_cfg.hidden_dim,
                vocab_size=model_cfg.vocab_size,
                lm_head_cfg=model_head_cfg,
            )


MODEL_SHELL_DICT = {"standard": model_shell.ModelShell, "byte_shell": ByteModelShell}


def build_model_shell(
    model_cfg: model_shell.ModelShellConfig | ByteShellConfig,
):
    """
    Given the model shell config, build it.
    Args:
        model_cfg: model_cfg
    Returns:
        model_shell: model_shell_instance
    """
    model_shell_type = model_cfg.model_shell_type
    # build the embedding model
    embedding_model = build_embedding_model(model_cfg=model_cfg)

    # build the core model
    core_model = build_core_model(model_cfg=model_cfg)

    # build the model head
    model_head = build_model_head(model_cfg=model_cfg, embedding_model=embedding_model)
    match model_shell_type:
        case "standard":
            return model_shell.ModelShell(
                embedding_model=embedding_model,
                core_model=core_model,
                model_head=model_head,
            )
        case "byte_shell":
            return ByteModelShell(
                embedding_model=embedding_model,
                core_model=core_model,
                model_head=model_head,
            )


def initialize_model(model_dict: dict):
    """
    Initialize the model given the configuration.
    Args:
        model_cfg: model_cfg
    Returns:
        model: model_instance
    """
    model_cfg = ModelShellConfigMap(**model_dict)
    model = build_model_shell(
        model_cfg=model_cfg,
    )
    # check if embedding model weights are to be shared with the model head
    if model_cfg.embedding_weight_tying:
        # share the weights between the token embeddings and the final
        # logit layer, following: https://paperswithcode.com/method/weight-tying
        model.embedding_model.token_embedder.weight = model.model_head.linear.weight

    # build the model shell

    return model
