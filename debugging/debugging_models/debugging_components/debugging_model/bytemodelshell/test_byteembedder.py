import pytest
import torch

from models.experimental.byte_level.embedding_model import ByteLevelEmbedder

def mock_inputids(embedderclass, model_cfg):
    """
    Initialize the input_ids tensor.
    """
    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['byte_vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['byte_context_window'])) # (batch_size, sequence_len, byte_context_window)

    ## build the embedder
    embedder = embedderclass(model_cfg)

    ## get the output
    res = embedder(input_ids)

    return res, embedder

def test_bytelevelembdder():
    """
    Test the embedding model.
    """
    model_cfg = {
        'embedder': {
            'tokenizer_type': 'gpt2', 
            'byte_tokenizer_type': 'bpe',
            'dataset_name': 'simple_en_wiki'
            }, 
        'hidden_dim': 768,
        'vocab_size': 50257, 
        'byte_vocab_size': 258, 
        'byte_context_window': 12,
        'byte_embedding_dim': 128, 
        'context_window': 512,
        'batch_size': 1
        }

    ## get the output
    res, embedder = mock_inputids(ByteLevelEmbedder, model_cfg)

    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])