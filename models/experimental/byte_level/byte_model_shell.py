"""
The standard Model Shell. It combines the embedding model,
core model and LM head.
"""

import torch

from models import core_models, embedding_models, model_heads
from models.model_shell import ModelShell


class ByteModelShell(ModelShell):
    """
    Slight deviation from the standard Model Shell to
    allow for a re-constructive auxiliary loss to the input.
    """

    def __init__(
        self,
        embedding_model: embedding_models.EmbedderInterface,
        core_model: core_models.GenericTransformer,
        model_head: model_heads.AutoregressiveLMHead,
        weight_init_func=None,
    ):
        super().__init__(
            embedding_model=embedding_model,
            core_model=core_model,
            model_head=model_head,
            weight_init_func=weight_init_func,
        )

    def forward(self, token_ids):
        """
        Forward pass with a re-constructive auxiliary loss.
        """
        # pass the token_ids through the embedding model
        # to get B, S, H (with pos encoding if necessary)
        x = self.embedding_model(token_ids)

        # calculate the reconstruction loss
        logits = self.model_head(x)[0]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), token_ids.view(-1), ignore_index=257
        )

        # pass the embeddings through the core model
        x = self.core_model(x)

        # pass the core model output through the model head
        x = self.model_head(x)[0]

        return x, loss
