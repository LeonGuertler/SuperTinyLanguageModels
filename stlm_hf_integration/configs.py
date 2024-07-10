"""Wrapper from huggingface"""

from transformers import PretrainedConfig


class STLMConfig(PretrainedConfig):
    model_type = "stlm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":
    stlmconfig = STLMConfig()
    stlmconfig.save_pretrained("stlm")
