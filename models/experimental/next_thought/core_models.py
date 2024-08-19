"""
The core next-thought model.
"""

from typing import Literal

import torch

from models.core_models import CoreModelConfig


class NextThoughtConfig(CoreModelConfig):
    """
    Next Thought configuration
    """

    core_model_type: Literal["next_thought"]
    latent_dim: int


class BaselineCoreModel(torch.nn.Module):
    """
    An extremely simplistic core model for
    next thought prediction.
    """

    def __init__(self, model_cfg: NextThoughtConfig):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=model_cfg.latent_dim,
                out_features=model_cfg.latent_dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=model_cfg.latent_dim,
                out_features=model_cfg.latent_dim,
            ),
        )

    def forward(self, x):
        """
        Pass an input through the model
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, H)
        """
        return self.model(x)


class Conv1dCoreModel(torch.nn.Module):
    """
    A core model for next thought prediction using Conv1d layers.
    """

    def __init__(self):
        super().__init__()

        # 4800
        self.conv1 = torch.nn.Linear(30, 30)
        self.conv2 = torch.nn.Linear(300, 300)
        self.conv3 = torch.nn.Linear(3, 3)

    def forward(self, x):
        """
        Pass an input through the model
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, H)
        """
        # B, 4800 -> B, 4800/30, 30
        x = x.view(x.size(0), 160, 30)
        x = self.conv1(x)
        x = x.view(x.size(0), 4800)

        # B, 4800 -> B, 4800/300, 300
        x = x.view(x.size(0), 16, 300)
        x = self.conv2(x)
        x = x.view(x.size(0), 4800)

        # B, 4800 -> B, 4800/3, 3
        x = x.view(x.size(0), 1600, 3)
        x = self.conv3(x)
        x = x.view(x.size(0), 4800)
        return x
