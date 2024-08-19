"""
A collection of different model heads.
"""

from typing import Literal

import pydantic
import torch

from models.components.layers.normalization import build_normalization


class LMHeadConfig(pydantic.BaseModel):
    """
    Head configuration
    """

    lm_head_type: str


class GenericLMHeadConfig(LMHeadConfig):
    """
    Language Model Head configuration
    """

    lm_head_type: Literal["generic"]
    normalization: str
    bias: bool


class HeadInterface(torch.nn.Module):
    """
    Interface for the head component of the model.
    """

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        This function should take the input tensor x as input,
        and return the output tensor.
        """
        raise NotImplementedError

    def inference(self, x):
        """
        Pass the input through the model, then
        Return the final token logits
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, V)
        """
        return self.forward(x)[0][:, -1, :]


class AutoregressiveLMHead(torch.nn.Module):
    """
    Generic autoregressive language model head.
    """

    def __init__(self, hidden_dim, vocab_size, lm_head_cfg: GenericLMHeadConfig):
        """
        Initialize the model.
        Args:
            hidden_dim: int
            vocab_size: int
            lm_head_cfg: LMHeadConfig
        """
        super().__init__()
        self.layer_norm = build_normalization(
            normalization_name=lm_head_cfg.normalization,
            dim=hidden_dim,
            bias=lm_head_cfg.bias,
        )
        self.linear = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=vocab_size,
            bias=lm_head_cfg.bias,
        )

    def forward(self, x):
        """
        Pass the input through the model.
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, V)
        """

        # apply layer norm
        x = self.layer_norm(x)

        # pass through the linear layer
        x = self.linear(x)

        return x, None
