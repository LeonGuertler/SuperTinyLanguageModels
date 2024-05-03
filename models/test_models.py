"""
A test function to test that the different models are loaded correctly.
"""

import io
import unittest.mock

# pylint: disable=unused-import
# pylint: disable=missing-function-docstring
import pytest
import torch

from models.build_models import build_model

sample_cfg = {
    "model_shell": {
        "shell_type": "autoregressive",
        "tokenizer": "bpe",
        "tokenizer_dataset_name": "debug",
        "vocab_size": 512,
        "context_window": 512,
    },
    "core_model": {
        "core_model_type": "modern",
        "hidden_dim": 512,
        "depth": 2,
        "num_heads": 8,
        "ffn_dim": 512,
        "ffn_activation": "gelu",
        "dropout": 0.1,
        "bias": False,
    },
    "trainer": {
        "dataset": "debug",
        "training": {
            "trainer": "base_trainer",
            "batch_size": 24,
            "gradient_accumulation_steps": 20,
            "max_iters": 5,
            "lr_decay_iters": 5,
            "warmup_iters": 2,
            "eval_interval": 5,
            "log_interval": 1,
            "eval_iters": 200,
            "checkpoint_interval": 1e9,
        },
        "optimizer": {
            "name": "nanoGPTadamW",
            "lr": 6e-4,
            "min_lr": 6e-5,
            "weight_decay": 1e-1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "decay_lr": True,
            "warmup_iters": 1000,
        },
        "scheduler": {"name": "cosine"},
        "dataloader": {"name": "standard"},
        "loss_fn": {"name": "cross_entropy"},
    },
}


def test_build_model():
    test_tokens = torch.randint(0, 512, (1, 512))
    test_string = "This is a test string"
    model = build_model(sample_cfg)

    # pass the tokens
    _ = model(test_tokens)

    # pass the string
    _ = model.inference(test_string)

    # save model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": None,
        "iter_num": 0,
        "config": sample_cfg,
    }

    # Create a virtual file
    with unittest.mock.patch("torch.save") as mock_save:
        mock_save.return_value = None
        mock_file = io.BytesIO()
        torch.save(checkpoint, mock_file)
        mock_file.seek(0)

        print("loading model from virtual file")
        # load model
        checkpoint = torch.load(mock_file)
        model = build_model(checkpoint=checkpoint)

    assert model is not None
