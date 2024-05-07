"""Contains the shell classes for the models."""

from torch import nn

from models.components.lm_heads import build_lm_head
from models.components.tokenizers import build_tokenizer
from models.shells.autoregressive_byte_model_shell import (
    AutoregressiveByteModelShell,
    ByteLevelProcessor,
)
from models.shells.autoregressive_model_shell import AutoregressiveModelShell


def build_shell(cfg, core_model):
    """
    Build the shell.
    """

    lm_head = build_lm_head(
        head_type=cfg["model_shell"]["lm_head"],
        hidden_dim=cfg["core_model"]["hidden_dim"],
        vocab_size=cfg["model_shell"]["vocab_size"],
    )
    token_embedder = nn.Embedding(
        num_embeddings=cfg["model_shell"]["vocab_size"],
        embedding_dim=cfg["model_shell"]["embedding_dim"],
    )

    if cfg["shell"]["type"] == "autoregressive":
        tokenizer = build_tokenizer(
            tokenizer_type=cfg["model_shell"]["tokenizer"],
            vocab_size=cfg["model_shell"]["vocab_size"],
            dataset_name=cfg["model_shell"]["tokenizer_dataset_name"],
        )
        return AutoregressiveModelShell(
            tokenizer=tokenizer,
            token_embedder=token_embedder,
            lm_head=lm_head,
            core_model=core_model,
        )
    elif cfg["shell"]["type"] == "autoregressive_byte":
        pooling_tokenizer = build_tokenizer(
            tokenizer_type=cfg["model_shell"]["pooling_tokenizer"],
            vocab_size=cfg["model_shell"]["vocab_size"],
            dataset_name=cfg["model_shell"]["tokenizer_dataset_name"],
        )
        byte_tokenizer = build_tokenizer(
            tokenizer_type=cfg["model_shell"]["byte_tokenizer"],
            vocab_size=cfg["model_shell"]["vocab_size"],
            dataset_name=cfg["model_shell"]["tokenizer_dataset_name"],
        )
        processor = ByteLevelProcessor(
            embedding_dim=cfg["model_shell"]["embedding_dim"],
            hidden_dim=cfg["core_model"]["hidden_dim"],
            pooling_tokenizer=pooling_tokenizer,
            byte_tokenizer=byte_tokenizer,
            token_embedder=token_embedder,
        )
        return AutoregressiveByteModelShell(
            processor=processor,
            token_embedder=token_embedder,
            lm_head=lm_head,
            core_model=core_model,
        )
    raise ValueError(f"Shell type {cfg['shell']['type']} not recognized.")
