"""Wrapper from huggingface"""

from stlm_modelling import ModelShell, build_model
from transformers import PreTrainedModel

from stlm_hf_integration.configs import STLMConfig


class STLMModel(PreTrainedModel):
    """Big model"""

    config_class = STLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.model: ModelShell = build_model(config)

    def forward(self, token_ids):
        return self.model(token_ids)
