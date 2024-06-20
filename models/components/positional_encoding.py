"""
A collection of positional encoding modules.
"""

import torch
import math


class LearnedPosEncoding(torch.nn.Module):
    """
    Basic learned positional encoding
    """

    def __init__(self, hidden_dim, context_window):
        super().__init__()
        self.pe = torch.nn.Embedding(
            num_embeddings=context_window, embedding_dim=hidden_dim
        )

    def forward(self, x):
        """
        Takes the input tensor and returns it positionally encoded.
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, H)
        """
        if len(x.shape) >= 2:
            return x + (self.pe(torch.arange(x.size(1), device=x.device)).unsqueeze(0))
        else:
            return x + self.pe(torch.arange(x.size(1), device=x.device))


class IdentityEncoding(torch.nn.Module):
    """
    In case RoPE is used, there is no need for an initial positional encoding.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Returns the input tensor as is.
        """
        return x

class SinCosPosEncoding(
    torch.nn.Module
):
    """SinCos encoding taken from:
    \\url{https://github.com/pytorch/examples/blob/main/word_language_model/model.py#L65}
    As used in the Vaiswani et al. paper..."""
    def __init__(self, hidden_dim, context_window):
        """Set up the pe buffer etc."""
        super().__init__()
        pe = torch.zeros(context_window, hidden_dim)
        position = torch.arange(0, context_window, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # pe has shape (1, S, H)

        self.pe = torch.nn.Parameter(pe) # hack for distributed data parallel
        self.pe.requires_grad = False

    def forward(self, x):
        """Add the pe to the input tensor."""
        # x of shape (B, S, H)
        return x + self.pe[:, :x.size(1)]


POS_ENCODING_DICT = {
    "learned": lambda dim, size, **_: LearnedPosEncoding(
        hidden_dim=dim, context_window=size
    ),
    "rope": lambda **_: IdentityEncoding(),
    "none": lambda **_: IdentityEncoding(),
    "sincos": lambda dim, size, **_: SinCosPosEncoding(
        hidden_dim=dim, context_window=size
    ),
}

def build_positional_encodings(model_cfg):
    """
    Given the positional encoding config, build it.
    Args:
        cfg: cfg
    Returns:
        positional_encodings: positional_encodings_instance
    """
    return POS_ENCODING_DICT[model_cfg["positional_encoding_type"]](
        dim=model_cfg["hidden_dim"], size=model_cfg["context_window"]
    )
