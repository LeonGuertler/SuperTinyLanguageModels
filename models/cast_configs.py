"""Pseudo Configs for Model Building/Casting"""

from typing import Literal

from models import core_models, embedding_models, model_heads, model_shell
from models.experimental import hugging_face
from models.experimental.next_thought import core_models as nt_core_models
from models.experimental.next_thought import embedding_models as nt_embedding_models


class ModelShellConfigMap(model_shell.ModelShellConfig):
    """Config for the standard model shell"""

    model_shell_type: Literal["standard"]
    core_model: (
        hugging_face.CoreModelConfig
        | nt_core_models.CoreModelConfig
        | core_models.CoreModelConfig
    )
    embedding_model: (
        nt_embedding_models.HierarchicalEncoderConfig
        | hugging_face.HFEmbedderConfig
        | embedding_models.GenericEmbedderConfig
    )
    model_head: hugging_face.HFLMHeadConfig | model_heads.LMHeadConfig
    hidden_dim: int
    context_window: int
    vocab_size: int
    embedding_weight_tying: bool
    positional_encoding_type: str
