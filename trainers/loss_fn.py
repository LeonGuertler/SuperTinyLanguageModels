"""Loss Functions for Training

Each loss function should take in output of a model and the target labels
and return the loss value. This need not be the logits."""

import torch


def cross_entropy_loss_fn(logits, y):
    """Cross Entropy Loss Function"""
    #print(logits.size())
    #input(y.size())
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    return torch.nn.functional.cross_entropy(logits, y, ignore_index=-1)


def compute_perplexity(logits, y, lengths: list[int]):
    """Compute perplexity

    The lengths are character lengths of the input sequences rather than
    of the tokenized sequences."""
    loss = torch.nn.functional.cross_entropy(
        logits, y, ignore_index=-1, reduction="sum"
    )
    return torch.exp(loss / sum(lengths))


def build_loss_fn(loss_fn_type: str):
    """Build the loss function"""
    if loss_fn_type == "cross_entropy":
        return cross_entropy_loss_fn
    raise ValueError(f"Loss function {loss_fn_type} not supported.")
