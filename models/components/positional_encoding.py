"""
A collection of positional encoding modules.
"""

import torch


class LearnedPosEncoding(torch.nn.Module):
    """
    Basic learned positional encoding
    """

    def __init__(self, hidden_dim, context_window):
        super().__init__()
        self.pe = torch.nn.Embedding(num_embeddings=context_window, embedding_dim=hidden_dim)

    def forward(self, x):
        """
        Takes the input tensor and returns it positionally encoded.
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, H)
        """
        # check device
        input(x.size())
        input(self.pe(torch.arange(x.size(1), device=x.device)).unsqueeze(0).to(x.device).size())

        if len(x.shape) >= 2:
            return x + (
                self.pe(torch.arange(x.size(1), device=x.device))
                .unsqueeze(0)
                .to(x.device)
            )
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



POS_ENCODING_DICT = {
    "learned": lambda dim, size: LearnedPosEncoding(
        hidden_dim=dim, 
        context_window=size
    ),
    "rope": lambda dim, size: IdentityEncoding(),
    "none": lambda dim, size: IdentityEncoding()
}


def build_positional_encodings(model_cfg):
    """
    Given the positional encoding config, build it.
    Args:
        cfg: cfg
    Returns:
        positional_encodings: positional_encodings_instance
    """
    return POS_ENCODING_DICT[
        model_cfg["positional_encoding_type"]](
            dim=model_cfg["hidden_dim"],
            size=model_cfg["context_window"]
        )