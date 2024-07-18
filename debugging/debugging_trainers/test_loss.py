"""Tests for trainers/loss_fn.py"""
import pytest
import torch

from trainers.loss_fn import compute_perplexity, cross_entropy_loss_fn, masked_cross_entropy_loss_fn, next_token_mlm_loss_fn

def test_compute_perplexity():
    input_batch = torch.randn(2, 16, 64)
    target_batch = torch.randint(0, 63, (2, 16))
    char_lengths = [16, 16]
    mask = torch.ones_like(target_batch).float()
    mask[0, 8:] = 0
    perp = compute_perplexity(input_batch, target_batch, char_lengths, mask=mask)
    assert perp > 0

def test_compute_perplexity_no_mask():
    input_batch = torch.randn(2, 16, 64)
    target_batch = torch.randint(0, 63, (2, 16))
    char_lengths = [16, 16]
    perp = compute_perplexity(input_batch, target_batch, char_lengths)
    assert perp > 0

def test_cross_entropy_loss_fn():
    logits = torch.randn(2, 16, 64)
    target = torch.randint(0, 63, (2, 16))
    loss = cross_entropy_loss_fn(logits, target)
    assert loss > 0

def test_masked_cross_entropy_loss_fn():
    logits = torch.randn(2, 16, 64)
    target = torch.randint(0, 63, (2, 16))
    mask = torch.ones_like(target).float()
    mask[0, 8:] = 0
    loss = masked_cross_entropy_loss_fn(logits, target, mask)
    assert loss > 0

def test_next_token_mlm_loss_fn():
    logits = torch.randn(2, 16, 64)
    target = torch.randint(0, 63, (2, 16))
    ## mask the target with long integers
    mask = torch.rand(target.size()) < 0.15
    loss = next_token_mlm_loss_fn(logits, (target, mask))
    assert loss > 0

