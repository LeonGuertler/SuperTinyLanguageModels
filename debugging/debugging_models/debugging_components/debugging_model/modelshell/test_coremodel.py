'''
Pytest for the core_model, one of the three that makes the model.
'''

import pytest
import torch

from models.core_models import GenericFFNSharedTransfomer, GenericTransformer
from models.experimental.hugging_face import HFTransformerCore
from models.experimental.next_thought.core_models import BaselineCoreModel


def mock_inputembeds(transformerclass, model_cfg):
    """
    Initialize the input_ids tensor.
    """
    ## prepare inputs
    if transformerclass == BaselineCoreModel:
        input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['latent_dim'])).float()
    else:
        input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])).float()
    
    ## build the core model
    core_model = transformerclass(model_cfg)
    
    ## get the output
    res = core_model(input_ids)

    return res


def test_generictransformer():
    '''
    Test the core model.
    '''

    model_cfg = {
        'core_model': {
            'num_layers': 8, 
            'ffn': {
                'ffn_type': 'swiglu', 
                'ffn_dim': 1320, 
                'normalization': 'rms_norm', 
                'bias': False
                }, 
            'attn': {
                'attn_type': 'generic', 
                'num_heads': 16, 
                'normalization': 'rms_norm', 
                'group_size': 4, 
                'bias': False, 
                'is_causal': True
                }
            }, 
        'hidden_dim': 512, 
        'context_window': 512, 
        'positional_encoding_type': 'rope',
        'batch_size': 1,
        'vocab_size': 50257,
        }
    
    ## get the output
    res = mock_inputembeds(GenericTransformer, model_cfg)

    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'],  model_cfg['context_window'], model_cfg['hidden_dim'])


def test_genericffnsharedtransfomer():
    '''
    Test the core model.
    '''
    model_cfg = {
        'core_model': {
            'num_layers': 10, 
            'ffn': {
                'ffn_type': 'swiglu', 
                'ffn_dim': 1536, 
                'normalization': 'rms_norm', 
                'bias': False
                }, 
            'attn': {
                'attn_type': 'generic', 
                'num_heads': 16, 
                'normalization': 'rms_norm', 
                'group_size': 4, 
                'bias': False, 
                'is_causal': True
                }
            }, 
        'hidden_dim': 512, 
        'context_window': 512,  
        'positional_encoding_type': 'rope',
        'batch_size': 1,
        'vocab_size': 50257,
        }

    ## get the output
    res = mock_inputembeds(GenericFFNSharedTransfomer, model_cfg)

    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])


def test_hftransformercore():
    '''
    Test the core model
    '''

    model_cfg = {
        'model_string': 'Qwen/Qwen2-0.5B', 
        'hidden_dim': 896, # Qwen2-0.5B mini has hidden_dim = 896
        'context_window': 512,
        'vocab_size': 151936, # Qwen2-0.5B mini has vocab_size = 151936
        'batch_size': 1
        }

    ## get the output
    res = mock_inputembeds(HFTransformerCore, model_cfg)


    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])


def test_baselinecoremodel():
    '''
    test the core model
    '''

    model_cfg = {
        'latent_dim': 4800,
        'batch_size': 1,
        'context_window': 64,
        'vocab_size': 258
    }

    ## get the output
    res = mock_inputembeds(BaselineCoreModel, model_cfg)

    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['latent_dim'])


## seems weird that the x inputs are just the batch and hidden dim. Need to double check.
# def test_conv1dcoremodel():
#     '''
#     test the core model
#     '''
#     model_cfg = {
#         None
#     }

#     batch_size, hidden_dim, vocab_size = 1, 4800, 258
#     input_ids = torch.randint(0, vocab_size, (batch_size, hidden_dim)).float()

#     core_model = Conv1dCoreModel(model_cfg)
#     assert core_model(input_ids).shape == (batch_size, hidden_dim)