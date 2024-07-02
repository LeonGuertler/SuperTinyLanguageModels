'''
The model_shell class object's forward function uses 3 different forward functions from the 3 different class objects that are passed to it.
These functions perform overarching tasks that are 1) embedder, 2) core model, 3) model head.

Generally...
...the embedder is a class object that generally takes in a tensor (B, T) and returns a tensor (B, T, H).
...the core model is a class object that takes in a tensor (B, T, H) and returns a tensor (B, T, H).
...the model head is a class object that takes in a tensor (B, T, H) and returns a tensor (B, T, V).


This pytest will ensure that these outputs are correctly shaped and are not nan. 
'''

import pytest
import torch

from models.core_models import GenericFFNSharedTransfomer, GenericTransformer
from models.embedding_models import GenericEmbedder
from models.experimental.byte_level.embedding_model import ByteLevelEmbedder
from models.experimental.byte_level.model_heads import ByteLevelDecoder
from models.experimental.hugging_face import HFEmbedder, HFLMHead, HFTransformerCore
from models.experimental.next_thought.embedding_models import HierarchicalEncoder
from models.experimental.next_thought.model_heads import VariableLengthLatentDecoder
from models.experimental.next_thought.core_models import BaselineCoreModel, Conv1dCoreModel
from models.model_heads import AutoregressiveLMHead



################################################### embedders ###################################################

def test_genericembedder():
    """
    Test the embedding model.
    """
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

    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window']))

    ## build the embedder
    embedder = GenericEmbedder(model_cfg)

    ## get the output
    res = embedder(input_ids)

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
    model_cfg = {
        'model_string': 'microsoft/Phi-3-mini-4k-instruct', 
        'hidden_dim': 3072, # Phi-3 mini has hidden_dim = 3072
        'context_window': 512,
        'vocab_size': 32064,
        'batch_size': 1
        }

    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window']))

    embedder = HFEmbedder(model_cfg)
    res = embedder(input_ids)
    
    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert embedder(input_ids).shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])
    

## Something weird with the HierarchicalEncoder's output shape. Need to double check.
def test_hierarchicalencoder():
    '''
    Test the embedding model.
    '''
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
    
    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window']))

    ## build the embedder    
    embedder = HierarchicalEncoder(model_cfg)

    ## get the output
    res = embedder(input_ids)

    ## 1. ensure the output is float32
    assert res.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(res).all()
    ## 3. ensure the output shape is correct
    assert res.shape == (model_cfg['batch_size'], model_cfg['latent_dim'])

################################################### coremodel ###################################################

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

    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])).float()
    
    ## build the core model
    core_model = GenericTransformer(model_cfg)
    
    ## get the output
    res = core_model(input_ids)

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

    ## prepare inputs
    batch_size, vocab_size = 1, 258
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])).float()

    ## build the core model
    core_model = GenericFFNSharedTransfomer(model_cfg)

    ## get the output
    res = core_model(input_ids)

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
        'model_string': 'microsoft/Phi-3-mini-4k-instruct', 
        'hidden_dim': 3072, # Phi-3 mini has hidden_dim = 3072
        'context_window': 512,
        'vocab_size': 32064, # Phi-3 mini has vocab_size = 32064
        'batch_size': 1
        }

    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])).float()

    ## build the core model
    core_model = HFTransformerCore(model_cfg)

    ## get the output
    res = core_model(input_ids)

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

    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['latent_dim'])).float()

    ## build the core model
    core_model = BaselineCoreModel(model_cfg)

    ## get the output
    res = core_model(input_ids)

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

################################################### modelheads ###################################################

def test_autoregressivelmhead():
    '''
    Test the model head. Typically goes from hidden_dim to vocab_size.
    '''
    model_cfg = {
        'hidden_dim': 64,
        'vocab_size': 258,
        'lm_head': {
            'normalization': 'rms_norm',
            'bias': False
        },
        'batch_size': 1,
        'context_window': 64
    }

    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])).float()

    ## build the model head
    model_head = AutoregressiveLMHead(model_cfg)

    ## get the output
    res = model_head(input_ids)
    assert len(res) == 2 ## ensure there are 2 values
    logits, _ = res

    ## 1. ensure the output is float32
    assert logits.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(logits).all()
    ## 3. ensure the output shape is correct
    assert logits.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['vocab_size']) ## ensure the shape of the result


def test_HFLMHead():
    '''
    Test the model head. Typically goes from hidden_dim to vocab_size.
    '''
    model_cfg = {
        'model_string': 'microsoft/Phi-3-mini-4k-instruct', 
        'hidden_dim': 3072, # Phi-3 mini has hidden_dim = 3072
        'context_window': 512,
        'vocab_size': 32064, # Phi-3 mini has vocab_size = 32064
        'batch_size': 1
        }

    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])).float()

    ## build the model head
    model_head = HFLMHead(model_cfg)

    ## get the output
    res = model_head(input_ids)
    assert len(model_head(input_ids)) == 2 ## ensure there are 2 values
    logits, _ = res

    ## 1. ensure the output is float32
    assert logits.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(logits).all()
    ## 3. ensure the output shape is correct
    assert logits.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['vocab_size'])


# def test_variablelengthlatentdecoder():
#     '''
#     Test the model head. Typically goes from hidden_dim to vocab_size.
#     '''
#     model_cfg = {
#         'lm_head': {
#             'latent_decoded_into': 16, #
#             'standard_ffn_block': { #
#                 'ffn_type': 'swiglu', 
#                 'ffn_dim': 1536, 
#                 'normalization': 'rms_norm', 
#                 'bias': False
#                 }, 
#             'standard_attn_block': { #
#                 'attn_type': 'generic', 
#                 'num_heads': 16, 
#                 'normalization': 'rms_norm', 
#                 'group_size': 4, 
#                 'bias': False, 
#                 'is_causal': True
#                 }
#             }, 
#         'latent_dim': 4800, #
#         'embedding_dim': 768, #
#         'hidden_dim': 768, 
#         'context_window': 512,#
#         'vocab_size': 50257, #
#         'batch_size': 1
#         }

#     input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['latent_dim'])).float()


#     cfg = {'model': {'core_model': {'core_model_type': 'next_thought_baseline'}, 'embedder': {'tokenizer_type': 'gpt2', 'embedding_model_type': 'hierarchical', 'dataset_name': 'simple_en_wiki', 'pooling_layers': 5, 'pooling_dims': [768, 1920, 1920, 1920, 4800], 'pct_pool_per_layer': [0.3, 0.5, 0.6, 0.6], 'num_heads': 12, 'context_window': 512, 'standard_ffn_block': {'ffn_type': 'swiglu', 'ffn_dim': 1536, 'normalization': 'rms_norm', 'bias': False}, 'standard_attn_block': {'attn_type': 'generic', 'num_heads': 16, 'normalization': 'rms_norm', 'group_size': 4, 'bias': False, 'is_causal': False}}, 'lm_head': {'lm_head_type': 'latent_2_seq', 'latent_decoded_into': 16, 'num_layers': 4, 'standard_ffn_block': {'ffn_type': 'swiglu', 'ffn_dim': 1536, 'normalization': 'rms_norm', 'bias': False}, 'standard_attn_block': {'attn_type': 'generic', 'num_heads': 16, 'normalization': 'rms_norm', 'group_size': 4, 'bias': False, 'is_causal': True}}, 'latent_dim': 4800, 'embedding_dim': 768, 'hidden_dim': 768, 'context_window': 512, 'vocab_size': 50257, 'model_shell_type': 'standard', 'embedding_weight_tying': False, 'positional_encoding_type': 'learned'}, 'trainer': {'dropout_scheduler': {'dropout_type': 'constant', 'dropout': 0.1, 'start_dropout_p': 0.0, 'end_dropout_p': 0.1, 'start_iter': 0, 'end_iter': 10000}, 'dataset': 'openhermes-2.5', 'training': {'trainer_type': 'base_trainer', 'batch_size': 24, 'gradient_accumulation_steps': 20, 'max_iters': 25000, 'lr_decay_iters': 25000, 'warmup_iters': 5000, 'eval_interval': 2000, 'log_interval': 10, 'eval_iters': 500, 'checkpoint_interval': 1000000000.0, 'run_profiler': False}, 'optimizer': {'name': 'nanoGPTadamW', 'lr': 0.0018, 'min_lr': 6e-05, 'weight_decay': 0.1, 'beta1': 0.9, 'beta2': 0.95, 'grad_clip': 1.0, 'decay_lr': True, 'warmup_iters': 5000}, 'lr_scheduler': {'name': 'cosine'}, 'dataloader': {'name': 'conversational'}, 'loss_fn': {'name': 'cross_entropy'}, 'eval': {'benchmarks': ['winograd', 'hellaswag', 'arc', 'mmlu', 'blimp'], 'num_samples': 5000, 'evaluator': 'mcq'}}, 'general': {'logging': {'wandb_log': True, 'wandb_project': 'SuperTinyLanguageModels'}, 'paths': {'output_dir': 'outputs', 'data_dir': 'data', 'checkpoint_dir': 'checkpoints'}, 'seed': 489, 'device': 'cuda'}}
#     model_cfg = cfg['model']
#     embedder = HierarchicalEncoder(model_cfg)


#     model_head = VariableLengthLatentDecoder(model_cfg, embedder)
#     assert len(model_head(input_ids)) == 2 ## ensure there are 2 values

#     res, _ = model_head(input_ids)
#     assert res.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['vocab_size'])