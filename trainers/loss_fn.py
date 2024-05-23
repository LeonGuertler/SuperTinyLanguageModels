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
    """
    Compute perplexity
    Args:
        logits: torch.tensor(B, S, H) or torch.tensor(B, S, S_c, H_c)
        y: torch.tensor(B, S) or torch.tensor(B, S, S_c)
        token_lengths: List[List[int]]
        char_lengths: List[int]
    Returns:
        perplexity: torch.tensor(1)
    """

    # check if logits is byte-level
    if len(logits.size()) > 3:
        B, S, S_c = y.size()
        logits = logits.view(B, S*S_c, -1)
        y = y.view(B, S*S_c)


    # B, S, H / B, S, 1
    # calculate non-reduced loss
    # flatten both
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    input(loss.size())
    # B, S, 1
    # unflatten
    loss = loss.view(B, S*S_c)
    input(token_lengths)
    input(char_lengths)

    total_loss = 0
    for i in range(B):
        for ii in range(S*S_c):
            total_loss += loss[i][ii] * token_lengths[i][ii]

    # multiply by the token lengths
    #loss = loss * torch.tensor(token_lengths).float()

    # sum and divide by character length
    loss = loss.sum() / torch.tensor(char_lengths).sum()

    return torch.exp(loss)



def build_loss_fn(loss_fn_type: str):
    """Build the loss function"""
    if loss_fn_type == "cross_entropy":
        return cross_entropy_loss_fn
    raise ValueError(f"Loss function {loss_fn_type} not supported.")