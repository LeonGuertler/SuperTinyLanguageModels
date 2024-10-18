"""
A collection of different model heads.
"""

import torch

from models.components.normalization import build_normalization


class AutoregressiveLMHead(torch.nn.Module):
    """
    Generic autoregressive language model head.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )
        self.linear = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["vocab_size"],
            bias=model_cfg["lm_head_bias"],
        )
        self.dropout = torch.nn.Dropout(
            p=model_cfg.get("lm_head_dropout", 0.0) # Default is no Dropout
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

        # apply dropout if necessary
        x = self.dropout(x)

        # pass through the linear layer
        x = self.linear(x)

        return x, None

    def inference(self, x):
        """
        Pass the input through the model, then
        Return the final token logits
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, V)
        """
        return self.forward(x[:, -1, :])[0]



class ClassificationLMHead(torch.nn.Module):
    """ TODO """

    def __init__(self, model_cfg):
        super().__init__()
        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head_normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head_bias"],
        )
        self.linear = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["lm_head_num_classes"],
            bias=model_cfg["lm_head_bias"],
        )
        self.dropout = torch.nn.Dropout(
            p=model_cfg.get("lm_head_dropout", 0.0) # Default is no Dropout
        )

    def forward(self, x):
        """
        Pass the input through the model.
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, V)
        """

        # only use the final token
        x = x[:, -1, :]

        # apply layer norm
        x = self.layer_norm(x)

        # apply dropout if necessary
        x = self.dropout(x)

        # pass through the linear layer
        x = self.linear(x)

        return x, None