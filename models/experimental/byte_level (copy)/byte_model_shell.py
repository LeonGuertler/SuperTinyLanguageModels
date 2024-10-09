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

        self.device = torch.device("cuda")  # Initialize to CPU or any default device

        # Initialize weights if necessary
        self._initialize_weights()

        # Print basic model statistics
        self._print_model_stats()


        # get loss hyperparameters
        self.target_chunk_len = model_cfg["target_chunk_len"]
        self.chunk_len_loss_weight = model_cfg["chunk_len_loss_weight"]
        self.max_chunk_length = model_cfg["max_chunk_length"]
        self.chunk_len_penalty = model_cfg["chunk_len_penalty"]



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
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Embedding Model Parameter Count: {count_parameters(self.embedding_model):,}")
        print(f"Core Model Parameter Count: {count_parameters(self.core_model):,}")
        print(f"Model Head Parameter Count: {count_parameters(self.model_head):,}")


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
        # Pass the token_ids through the embedding model
        # to get embeddings and target_ids (B, S, H) and (B, S)
        embeddings, target_ids, avg_chunk_len, target_mask, chunk_len_loss = self.embedding_model(token_ids) 
        # print(embeddings.size())
        # print(f"Embeddings: {embeddings.size()}")
        # print(f"target_ids: {target_ids.size()}")
        # print(f"Avg. chunk len: {avg_chunk_len}")
        # exit()
        
        # Pass the embeddings through the core model
        core_output = self.core_model(embeddings)

        # Pass the core model output through the model head to get logits (B, S, V)
        logits = self.model_head(core_output, target_ids)

        # input(logits.size()) # [2, 3456, 6, 259]
        # input(target_ids.size()) # [2, 3456, 6]
        # input(avg_chunk_len)
        # exit()


        # Compute the loss, ignoring pad tokens
        loss = cross_entropy_loss_fn(
            logits=logits,
            y=target_ids.long(),
            ignore_index=self.embedding_model.pad_token_id
        )

        # Aux loss 1: Target Chunk length
        chunk_loss = chunk_len_loss #* (avg_chunk_len - self.target_chunk_len) ** 2

        # Aux loss 2: Max Chunk length
        over_length = torch.clamp(
            avg_chunk_len-self.max_chunk_length,
            min=0
        )
        length_loss = torch.sum(over_length)


        total_loss = loss + \
            self.chunk_len_loss_weight * chunk_loss #+ \
        #self.chunk_len_penalty * length_loss

        #print(f"Total Loss: {total_loss}, Chunk Loss: {chunk_loss}, BCE Loss: {loss}")

        additional_info_dict = {
            "average_chunk_length": avg_chunk_len.item(),
            "chunk_len_loss": self.chunk_len_loss_weight*chunk_loss,
            "chunk_len_penalty_loss": self.chunk_len_penalty*length_loss,
            "BCE-loss": loss,
            "total-loss": total_loss,
        }

        return core_output, total_loss, additional_info_dict
