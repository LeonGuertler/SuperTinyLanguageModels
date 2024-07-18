import pytest
import torch

from models.experimental.byte_level.model_heads import ByteLevelDecoder


def mock_hiddenstates(modelheadclass, model_cfg):
    """
    Initialize the input_ids tensor.
    """
    ## prepare inputs
    input_ids = torch.randint(0, model_cfg['byte_vocab_size'], (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['hidden_dim'])).float()

    ## build the model head
    model_head = ByteLevelDecoder(model_cfg)

    ## get the output
    res = model_head(input_ids)

    return res

def test_byteleveldecoder():
    '''
    Test the model head. Typically goes from hidden_dim to vocab_size.
    '''
    model_cfg = {
        'byte_vocab_size': 258,
        'byte_embedding_dim': 128,
        'byte_context_window': 12,
        'hidden_dim': 768,
        'batch_size': 1,
        'context_window': 512
    }

    ## get the output
    res = mock_hiddenstates(ByteLevelDecoder, model_cfg)
    assert len(res) == 2 ## ensure there are 2 values
    logits, _ = res

    ## 1. ensure the output is float32
    assert logits.dtype == torch.float32
    ## 2. ensure the output is not nan
    assert not torch.isnan(logits).all()
    ## 3. ensure the output shape is correct
    assert logits.shape == (model_cfg['batch_size'], model_cfg['context_window'], model_cfg['byte_context_window'], model_cfg['byte_vocab_size'])
