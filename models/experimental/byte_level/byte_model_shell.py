"""
The standard Model Shell. It combines the embedding model,
core model and LM head.
"""

import torch

from models import core_models, embedding_models, model_heads
from models.model_shell import ModelShell 


class ByteEncModelShell(ModelShell):
    """
    A model shell for inference using the 
    byte-level encoder.
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
    @torch.no_grad()
    def inference(self, model_input):
        """
        Takes a string as input, and returns 
        the decoded model output. The actual decoing
        should happen in the decoding generator.
        Args:
            model_input: str
        Returns:
            logits: torch.tensor(B, S, V)
        """
        byte_input = self.embedding_model.tokenize_input(
            model_input, 
            trunate=True, 
            add_eot=False, 
            return_high_level=False
        )

        # convert to tensor
        x = torch.tensor(byte_input, device=self.device, dtype=torch.long).unsqueeze(0)

        # forward pass to logits
        x = self.forward(x)

        return x

        # get the per gpt2 token logits by adding the word level ones
        #logits = torch.sum(x, dim=-1)

        #return logits



class ByteModelShellAuxLoss(ModelShell):
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
            logits.view(-1, logits.size(-1)), 
            token_ids.view(-1), 
            ignore_index=257
        )

        # pass the embeddings through the core model
        x = self.core_model(x)

        # pass the core model output through the model head
        x = self.model_head(x)[0]

        return x, loss 
    


class ByteVAEShell(ModelShell):
    """
    Turn into a simple VAE to test autoencoder capabilities.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, token_ids):
        """
        Forward pass with 10% noise
        """
        x = self.embedding_model(token_ids)

        # add noise to 10% of the data
        noise = torch.randn_like(x)+0.5
        mask = torch.rand_like(x) < 0.1  # 10% mask
        x = torch.where(mask, x * noise, x)

        # pass through head
        x = self.model_head(x)[0]

        return x, None