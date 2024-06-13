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


def compute_perplexity(log_likelihood, char_lengths, mask=None):
    """
    Compute perplexity.
    
    Args:
        log_likelihood: torch.tensor(B, S)
        char_lengths: List[int]
        mask: Optional[torch.tensor(B, S)]
        
    Returns:
        perplexity: torch.tensor(1)
    """

    # Convert to CPU if necessary
    log_likelihood = log_likelihood.cpu()
    if mask is not None:
        mask = mask.cpu()

    # Ensure mask is applied if provided
    if mask is not None:
        log_likelihood = log_likelihood * mask

    # Sum log likelihoods across the sequence dimension
    summed_log_likelihood = log_likelihood.sum(dim=-1)

    # Adjust for character lengths
    normalized_log_likelihood = summed_log_likelihood / torch.tensor(char_lengths).view(-1)

    # Compute the average log likelihood
    avg_log_likelihood = normalized_log_likelihood.mean()

    # Calculate perplexity
    perplexity = torch.exp(-avg_log_likelihood)

    return perplexity.item()



def build_loss_fn(loss_fn_type: str):
    """Build the loss function"""
    if loss_fn_type == "cross_entropy":
        return cross_entropy_loss_fn
    raise ValueError(f"Loss function {loss_fn_type} not supported.")
