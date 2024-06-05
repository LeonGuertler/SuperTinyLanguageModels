"""Tests for trainers/loss_fn.py"""
import pytest
import torch

from trainers.loss_fn import compute_perplexity

def test_compute_perplexity():
    input_batch = torch.randn(2, 16, 64)
    target_batch = torch.randint(0, 63, (2, 16))
    char_lengths = [16, 16]
    mask = torch.ones_like(target_batch).float()
    mask[0, 8:] = 0
    perp = compute_perplexity(input_batch, target_batch, char_lengths, mask=mask)
    assert perp > 0
