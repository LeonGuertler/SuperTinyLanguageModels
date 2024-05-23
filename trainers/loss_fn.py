"""Loss Functions for Training

Each loss function should take in output of a model and the target labels
and return the loss value. This need not be the logits."""

import torch


def cross_entropy_loss_fn(logits, y, mask=None):
    """Cross Entropy Loss Function"""

    # if there is an extra dimension, flatten first (byte-level-models)
    if len(logits.size()) > 2:
        B, S, S_c = y.size()
        logits = logits.view(B, S*S_c, -1)
        y = y.view(B, S*S_c)
    if mask is not None:
        logits = logits[mask]
        y = y[mask]
    else:
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

def compute_perplexity(logits, y, token_lengths, char_lengths, mask=None):
    """Compute perplexity"""
    input(logits.size())
    input(y.size())
    input(token_lengths.size())
    input(char_lengths.size())
    # get token level loss (masking as necessary)

    loss = torch.nn.functional.cross
    loss = cross_entropy_loss_fn(logits, y, mask=mask)
    return torch.exp(loss)


#def compute_perplexity(logits, y, lengths: list[int]):
#    """Compute perplexity

#    The lengths are character lengths of the input sequences rather than
#    of the tokenized sequences."""
#    loss = torch.nn.functional.cross_entropy(
#        logits, y, ignore_index=-1, reduction="sum"
#    )
#    return torch.exp(loss / sum(lengths))


def build_loss_fn(loss_fn_type: str):
    """Build the loss function"""
    if loss_fn_type == "cross_entropy":
        return cross_entropy_loss_fn
    raise ValueError(f"Loss function {loss_fn_type} not supported.")