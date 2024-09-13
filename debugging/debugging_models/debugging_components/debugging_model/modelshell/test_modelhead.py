'''
Pytest for the core_model, one of the three that makes the model.
'''

import pytest
import torch

from models.experimental.hugging_face import HFLMHead
from models.model_heads import AutoregressiveLMHead


def mock_hiddenstates(modelheadclass, model_cfg):
    """
    Initialize the input_ids tensor.
    """
    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])).float()

    ## build the model head
    model_head = modelheadclass(model_cfg)

    ## get the output
    res = model_head(input_ids)

    return res


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

    ## get the output
    res = mock_hiddenstates(AutoregressiveLMHead, model_cfg)
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
        'model_string': 'Qwen/Qwen2-0.5B', 
        'hidden_dim': 896, # Qwen2-0.5B mini has hidden_dim = 896
        'context_window': 512,
        'vocab_size': 151936, # Qwen2-0.5B mini has vocab_size = 151936
        'batch_size': 1
        }

    ## get the output
    res = mock_hiddenstates(HFLMHead, model_cfg)
    assert len(res) == 2 ## ensure there are 2 values
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