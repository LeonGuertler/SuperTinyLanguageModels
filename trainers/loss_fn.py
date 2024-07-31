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


def compute_perplexity(logits, y, char_lengths, mask=None):
    """
    Compute perplexity
    Args:
        logits: torch.tensor(B, S, H) or torch.tensor(B, S, S_c, H_c)
        y: torch.tensor(B, S) or torch.tensor(B, S, S_c)
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
    loss = loss * mask / torch.tensor(char_lengths).view(-1, 1)
    loss = loss.sum(dim=-1)

    return (torch.exp(loss)).mean().item()


def build_loss_fn(loss_fn_type: str):
    """Build the loss function"""
    if loss_fn_type == "cross_entropy":
        return cross_entropy_loss_fn
    raise ValueError(f"Loss function {loss_fn_type} not supported.")


def distillation_loss_fn(student_logits, teacher_logits, temperature=1.0, type='forward'):
    """
    Compute the distillation loss between student and teacher logits.
    Args:
        student_logits: torch.tensor(B, S, V)
        teacher_logits: torch.tensor(B, S, V)
        temperature: float
    Returns:
        distillation_loss: torch.tensor(1)
    """

    # flatten both
    student_logits = student_logits.view(-1, student_logits.size(-1))
    teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

    if type == 'forward':
        input_logits = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        target_logits = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    elif type == 'reverse':
        input_logits = torch.nn.functional.log_softmax(teacher_logits / temperature, dim=-1)
        target_logits = torch.nn.functional.softmax(student_logits / temperature, dim=-1)
    else:
        raise ValueError(f"Type {type} not supported.")
    
    # calculate KL Divergence
    distillation_loss = torch.nn.functional.kl_div(
        input_logits,
        target_logits,
        reduction="batchmean",
        log_target=False
    )

    ## return the distillation loss multiplied by the temperature squared as per the paper - https://arxiv.org/abs/1503.02531
    ## "Since the magnitudes of the gradients produced by the soft targets scale as 1/T^2
    ## it is important to multiply them by T 2 when using both hard and soft targets.""
    return distillation_loss * temperature ** 2