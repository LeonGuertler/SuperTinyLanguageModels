'''
Pytest for the embedding models, one of the three that makes the model.
'''

import pytest
import torch

from models.embedding_models import GenericEmbedder
from models.experimental.hugging_face import HFEmbedder
from models.experimental.next_thought.embedding_models import HierarchicalEncoder

def mock_inputids(embedderclass, model_cfg):
    """
    Initialize the input_ids tensor.
    """
    ## prepare mock inputs - shape (batch_size, context_window)
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window']))
    
    ## build the embedder
    embedder = embedderclass(model_cfg)

    ## get the output
    res = embedder(input_ids)

    return res, embedder

def test_genericembedder():
    """
    Test the embedding model.
    """

    ## prepare the mock config
    model_cfg = {
        "embedder": {
            "tokenizer_type": "gpt2",
            "embedding_model_type": "generic",
            "dataset_name": "stlm"
        },
        "hidden_dim": 64,
        "context_window": 64,
        "vocab_size": 50257,
        "positional_encoding_type": "rope",
        'batch_size': 1
    }

    ## get the output
    res, embedder = mock_inputids(GenericEmbedder, model_cfg)

    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'], model_cfg['context_window'], embedder.token_embedder.embedding_dim)


def test_hfembedder():
    '''
    Test the embedding model.
    '''

    ## prepare the mock config
    model_cfg = {
        'model_string': 'Qwen/Qwen2-0.5B', 
        'hidden_dim': 896, # Qwen2-0.5B mini has hidden_dim = 896
        'context_window': 512,
        'vocab_size': 151936, # Qwen2-0.5B mini has vocab_size = 151936
        'batch_size': 1
        }

    ## get the output
    res, embedder = mock_inputids(HFEmbedder, model_cfg)
    
    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])
    

## Something weird with the HierarchicalEncoder's output shape. Need to double check.
def test_hierarchicalencoder():
    '''
    Test the embedding model.
    '''

    ## prepare the mock config
    model_cfg = {
        'embedder': {
            'tokenizer_type': 'gpt2',
            'dataset_name': 'simple_en_wiki',
            'pooling_dims': [768, 1920, 1920, 1920, 4800],
            'pct_pool_per_layer': [0.3, 0.5, 0.6, 0.6],
            'context_window': 512,
            'standard_ffn_block': {
                'ffn_type': 'swiglu', 
                'ffn_dim': 1536, 
                'normalization': 'rms_norm', 
                'bias': False
                }, 
            'standard_attn_block': {
                'attn_type': 'generic', 
                'num_heads': 16, 
                'normalization': 'rms_norm', 
                'group_size': 4, 
                'bias': False, 
                'is_causal': False
                }
            },
        'latent_dim': 4800, 
        'hidden_dim': 768,
        'context_window': 512,
        'vocab_size': 50257,
        'positional_encoding_type': 'learned',
        'batch_size': 1
        }
    
    ## get the output
    res, embedder = mock_inputids(HierarchicalEncoder, model_cfg)

    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'], model_cfg['latent_dim'])
