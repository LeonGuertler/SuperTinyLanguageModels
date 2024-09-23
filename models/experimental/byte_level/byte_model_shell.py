"""
The standard Model Shell. It combines the embedding model,
core model and LM head.
"""

import torch

from models import core_models, embedding_models, model_heads
from models.model_shell import ModelShell 

import time 

def cross_entropy_loss_fn(logits, y, ignore_index=-1):
    """
    Cross Entropy Loss Function that ignores specified index.

    Args:
        logits (Tensor): The output logits from the model (B, S, V).
        y (Tensor): The target token IDs (B, S).
        ignore_index (int): The index to ignore in the loss computation.

    Returns:
        Tensor: The computed cross entropy loss.
    """
    logits = logits.view(-1, logits.size(-1))  # (B*S, V)
    y = y.view(-1)  # (B*S,)
    #print(logits.size(), y.size())
    return torch.nn.functional.cross_entropy(logits, y, ignore_index=ignore_index)

class ByteAutoencoderModelShell(torch.nn.Module):
    """
    Unify the embedding model, core model, and LM head 
    into a single object; initializes the weights
    and prints basic model statistics.
    """

    def __init__(
        self,
        model_cfg,
        embedding_model,
        core_model,
        model_head,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head

        self.device = torch.device("cpu")  # Initialize to CPU or any default device

        # Initialize weights if necessary
        self._initialize_weights()

        # Print basic model statistics
        self._print_model_stats()

    def _initialize_weights(self):
        """
        Initialize weights of the model components.
        """
        # Example initialization (modify as needed)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def _print_model_stats(self):
        """
        Print basic statistics about the model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")

    # Override the `to` method to set the device attribute
    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)

    def forward(self, token_ids):
        """
        The default forward pass is used for training and
        accepts the token_ids as input.

        Args:
            token_ids (Tensor): Input token IDs (B, S).

        Returns:
            Tuple[Tensor, Tensor]: The core model output and the loss.
        """
        #print(f"\n\n")
        #t0 = time.time()
        # Pass the token_ids through the embedding model
        # to get embeddings and target_ids (B, S, H) and (B, S)
        embeddings, target_ids, chunk_len = self.embedding_model(token_ids)
        
        #print(f"Embedding: {time.time() - t0:.5f} Seconds")
        # Pass the embeddings through the core model
        core_output = self.core_model(embeddings)
        #to = time.time()

        # Pass the core model output through the model head to get logits (B, S, V)
        logits = self.model_head(core_output)
        #print(f"LM Head: {time.time() - t0:.5f} Seconds")
        # Compute the loss, ignoring pad tokens
        loss = cross_entropy_loss_fn(
            logits=logits,
            y=target_ids.long(),
            ignore_index=self.embedding_model.pad_token_id
        ) #- 0.2*chunk_len
        chunk_loss = -0.002*chunk_len 

        total_loss = loss + chunk_loss

        print(f"Total Loss: {total_loss}, Chunk Loss: {chunk_loss}, BCE Loss: {loss}")

        return core_output, total_loss
