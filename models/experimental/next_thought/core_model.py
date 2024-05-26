"""
The core next-thought model.
"""
import torch 



class BaselineCoreModel(torch.nn.Module):
    """
    An extremely simplistic core model for 
    next thought prediction.
    """
    def __init__(self, model_cfg):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=model_cfg["latent_dim"],
                out_features=model_cfg["latent_dim"],
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=model_cfg["latent_dim"],
                out_features=model_cfg["latent_dim"],
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
    