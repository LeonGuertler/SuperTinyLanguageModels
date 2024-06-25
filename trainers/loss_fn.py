"""Loss Functions for Training

Each loss function should take in output of a model and the target labels
and return the loss value. This need not be the logits."""

import time

import torch


def masked_cross_entropy_loss_fn(logits, y, mask=None):
    """Cross Entropy Loss Function"""
    # mask the pad token from y 
    pad_token_id = 257
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    #return torch.nn.functional.cross_entropy(logits, y, weight=mask, ignore_index=-1)
    return torch.nn.functional.cross_entropy(logits, y, ignore_index=pad_token_id)

def cross_entropy_loss_fn(logits, y, mask=None):
    """Cross Entropy Loss Function"""
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    return torch.nn.functional.cross_entropy(logits, y, ignore_index=-1)


def next_token_mlm_loss_fn(logits, y_mask, masked_loss=True):
    """
    Using the mask to extract the masked tokens, calculate the next-token
    cross-entropy-loss. This was proposed in https://arxiv.org/abs/2404.05961
    to train document embedding models.
    """
    y, mask = y_mask
    if masked_loss:
        logits = logits[mask]
        y = y[mask]

    return cross_entropy_loss_fn(logits, y)



def build_loss_fn(loss_fn_type: str):
    """Build the loss function"""
    if loss_fn_type == "cross_entropy":
        return cross_entropy_loss_fn
    raise ValueError(f"Loss function {loss_fn_type} not supported.")
