"""Loss Functions for Training

Each loss function should take in output of a model and the target labels
and return the loss value. This need not be the logits."""

import torch
import time 

def masked_cross_entropy_loss_fn(logits, y, mask=None):
    """Cross Entropy Loss Function"""
    # mask the pad token from y 
    pad_token_id = 257
    mask = y != pad_token_id
    mask = mask.to(torch.int32)

    logits = mask * logits
    y = mask * y

    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    #return torch.nn.functional.cross_entropy(logits, y, weight=mask, ignore_index=-1)
    return torch.nn.functional.cross_entropy(logits, y, ignore_index=-1)

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

    # pull everything onto cpu
    logits = logits.cpu()
    y = y.cpu()
    if mask is not None:
        mask = mask.cpu()
    
    # check if logits is byte-level
    if len(logits.size()) > 3:
        B, S, S_c = y.size()
        seq_len = S * S_c
        logits = logits.view(B, seq_len, -1)
        y = y.view(B, seq_len)
    else:
        B, seq_len = y.size()
        


    # B, S, H / B, S, 1
    # calculate non-reduced loss
    # flatten both
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    # B, S, 1
    # unflatten
    loss = loss.view(B, seq_len)

    total_loss = 0
    for i in range(B):
        # mask and multiply
        if mask is not None:
            total_loss += (loss[i] * torch.tensor(token_lengths[i]).float())[mask[i]].sum()
        else:
            total_loss += (loss[i] * torch.tensor(token_lengths[i]).float()).sum()


    # sum and divide by character length
    loss = loss.sum() / torch.tensor(char_lengths).sum()

    return torch.exp(loss)



def build_loss_fn(loss_fn_type: str):
    """Build the loss function"""
    if loss_fn_type == "cross_entropy":
        return cross_entropy_loss_fn
    raise ValueError(f"Loss function {loss_fn_type} not supported.")