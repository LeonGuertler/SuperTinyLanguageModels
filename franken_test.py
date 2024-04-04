"""
Non-optimal testing structure by simply creating a bunch of testing functions
for everything that are called one by one.
"""

import pytest
import torch

"""
Test the individual layers. Specifically:
    - LayerNorm
    - CausalSelfAttention
    - FFN
"""
from models.layers import FFN, CausalSelfAttention, LayerNorm


def test_layer_norm():
    """
    Test the LayerNorm module.
    """
    # create a layer norm module
    ln = LayerNorm(10, bias=True)

    # create a random tensor
    x = torch.randn(10)

    # forward pass
    output = ln(x)

    # check the output shape
    assert output.shape == x.shape


def test_causal_self_attention():
    """
    Test the CausalSelfAttention module.
    """
    # create a causal self attention module
    csa = CausalSelfAttention(hidden_dim=10, num_heads=2, bias=True, dropout=0.1)

    # create a random tensor
    x = torch.randn(10, 10, 10)

    # forward pass
    output = csa(x)

    # check the output shape
    assert output.shape == x.shape


def test_ffn():
    """
    Test the FFN module.
    """
    # create a FFN module
    ffn = FFN(hidden_dim=10, ffn_dim=20, bias=True, dropout=0.1)

    # create a random tensor
    x = torch.randn(10, 10, 10)

    # forward pass
    output = ffn(x)

    # check the output shape
    assert output.shape == x.shape


"""
Test the embedding module.
"""
from models.embedding import BaselineEmbedder


def test_baseline_embedder():
    """
    Test the BaselineEmbedder module.
    """
    # create a baseline embedder
    be = BaselineEmbedder(hidden_dim=10, context_window=10, vocab_size=100_000)

    # create a random tensor
    x = "Hello World"

    # inidivdually encode
    token_ids_batch, attention_mask_1 = be.tokenize_text(x, pad_truncate=False)

    # embed
    x_1 = be.embedding(token_ids_batch)

    # full forward encode
    x_2, attention_mask_2 = be(x, pad_truncate=False)

    # check the output shape
    assert x_1.shape == x_2.shape

    # check if attention_masks are either both None or the same
    assert (attention_mask_1 is None and attention_mask_2 is None) or (
        attention_mask_1 == attention_mask_2
    )

    # decode tokens to confirm the output is the same as the input
    # print(token_ids_batch)
    assert be.tokenizer.decode_batch(token_ids_batch.tolist())[0] == x

    # test that the padding is working as expected
    x = [
        "Hello World",
        "Hello World, this is a test of the emergency broadcast system. This is only a test.",
    ]

    x_2, attention_mask_2 = be(
        x,
        pad_truncate=True,
    )

    # check the output shape
    assert x_2.shape[1] == 10


"""
Test the model builder and baseline model
"""
from models.baseline import BaseGPT
from models.build_models import build_model
from models.utils import print_model_stats


def test_build_model():
    # create mock model dict containing
    # hiddem_dim, bias, dropout, ffn_dim, num_heads, depth, vocab_size, context_window
    model_dict = {
        "model": "baseline",
        "hidden_dim": 10,
        "bias": True,
        "dropout": 0.1,
        "ffn_dim": 20,
        "num_heads": 2,
        "depth": 2,
        "vocab_size": 50256,
        "context_window": 10,
        "shared_embedding": True,
    }

    # create a model
    model = build_model(
        cfg=model_dict,
    )

    # test forward pass from token_ids
    x = torch.randint(0, 100, (10, 10))
    output = model(x)

    # check the output shape
    assert output.shape[:2] == x.shape
    assert output.size(-1) == model_dict["vocab_size"]

    # test with encoded string
    x = [
        "Hello World",
        "Hello World, this is a test of the emergency broadcast system. This is only a test.",
    ]

    model.inference(x)

    # x, _ = model.embedder(x, pad_truncate=True)
    # print('encoded successfullyl')
    # model(x)
    # print('model did it')


"""
Try loading the debug dataset
"""
from trainers.utils import load_data


def test_load_data():
    # load the debug dataset
    dataset = load_data(dataset_name="debug")


"""
Test the individual build functions
"""
from trainers.build_trainers import (
    build_dataloader,
    build_loss_fn,
    build_optimizer,
    build_scheduler,
)

# load the config
test_dict = {
    "general": {
        "logging": {"wandb_log": False, "wandb_project": "TinyUniverse"},
        "paths": {
            "output_dir": "outputs",
            "data_path": "data",
            "checkpoint_dir": "checkpoints",
        },
        "seed": 489,
        "device": "cuda",
    },
    "model": {
        "model": "baseline",
        "tokenizer": "gpt2",
        "context_window": 512,
        "vocab_size": 50304,
        "depth": 6,
        "hidden_dim": 64,
        "num_heads": 8,
        "ffn_dim": 256,
        "dropout": 0.0,
        "bias": False,
    },
    "trainer": {
        "dataset": "debug",
        "training": {
            "trainer": "base_profiler",
            "batch_size": 24,
            "gradient_accumulation_steps": 20,
            "max_iters": 1000,
            "lr_decay_iters": 1000,
            "warmup_iters": 10,
            "eval_interval": 500,
            "log_interval": 1,
            "eval_iters": 200,
            "checkpoint_interval": 500,
        },
        "optimizer": {
            "name": "nanoGPTadamW",
            "lr": 0.0006,
            "min_lr": 6e-05,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "decay_lr": True,
            "warmup_iters": 5000,
        },
        "scheduler": {"name": "cosine"},
        "dataloader": {"name": "standard"},
        "loss_fn": {"name": "cross_entropy"},
    },
}

# create necessary folders
from trainers.utils import create_folder_structure

# create the folder structure
create_folder_structure(path_config=test_dict["general"]["paths"])
# convert test dict to omegaconf
from omegaconf import OmegaConf

test_dict = OmegaConf.create(test_dict)


# this test will only run if device has cuda enabled


def test_build_optimizer():
    """
    Test the build_optimizer function.
    """
    model = BaseGPT(cfg=test_dict["model"])
    model.to("cuda")
    # build the optimizer
    optimizer = build_optimizer(
        model=model, optimizer_config=test_dict["trainer"]["optimizer"]
    )

    # check the optimizer type
    assert isinstance(optimizer, torch.optim.AdamW)


def test_build_scheduler():
    """
    Test the build_scheduler function.
    """
    # build the scheduler
    scheduler = build_scheduler(trainer_cfg=test_dict["trainer"])


def test_build_dataloader():
    """
    Test the build_dataloader function.
    """
    # build the dataloader
    dataloader = build_dataloader(cfg=test_dict)


def test_build_loss_fn():
    """
    Test the build_loss_fn function.
    """
    # build the loss function
    loss_fn = build_loss_fn(loss_fn_name=test_dict["trainer"]["loss_fn"]["name"])

    # create a random tensor
    x = torch.randn(10, 10, 10)
    y = torch.randint(0, 10, (10, 10))

    # forward pass
    output = loss_fn(x, y)


from trainers.build_trainers import build_trainer


def test_build_trainer():
    """
    Test the build_trainer function.
    """
    # build the trainer
    trainer = build_trainer(
        cfg=test_dict,
    )


"""
Do a small test training run
"""


def test_training_run():
    # load the relevant trainer
    trainer = build_trainer(
        cfg=test_dict,
    )
    # preprocess the training data
    trainer.preprocess_data()

    # train the model
    trainer.train()


"""
test the profiler 
"""
from trainers.base_profiler import BaseProfiler
# load the config

def test_training_run():
    test_dict = {
        "general": {
            "logging": {
                "wandb_log": False, 
                "wandb_project": "TinyUniverse",
                
            },
            "paths": {
                "output_dir": "outputs",
                "data_path": "data",
                "checkpoint_dir": "checkpoints",
            },
            "seed": 489,
            "device": "cuda",
        },
        "model": {
            "model": "baseline",
            "tokenizer": "gpt2",
            "context_window": 512,
            "vocab_size": 50304,
            "depth": 6,
            "hidden_dim": 64,
            "num_heads": 8,
            "ffn_dim": 256,
            "dropout": 0.0,
            "bias": False,
        },
        "trainer": {
            "dataset": "debug",
            "training": {
                "trainer": "base_profiler",
                "batch_size": 24,
                "gradient_accumulation_steps": 20,
                "max_iters": 1000,
                "lr_decay_iters": 1000,
                "warmup_iters": 10,
                "eval_interval": 500,
                "log_interval": 1,
                "eval_iters": 200,
                "checkpoint_interval": 100,
            },
            "optimizer": {
                "name": "nanoGPTadamW",
                "lr": 0.0006,
                "min_lr": 6e-05,
                "weight_decay": 0.1,
                "beta1": 0.9,
                "beta2": 0.95,
                "grad_clip": 1.0,
                "decay_lr": True,
                "warmup_iters": 5000,
            },
            "scheduler": {"name": "cosine"},
            "dataloader": {"name": "standard"},
            "loss_fn": {"name": "cross_entropy"},
        },
    }

    test_dict = OmegaConf.create(test_dict)
    # load the relevant trainer
    trainer = build_trainer(
        cfg=test_dict,
    )
    # preprocess the training data
    trainer.preprocess_data()

    # train the model
    trainer.train()