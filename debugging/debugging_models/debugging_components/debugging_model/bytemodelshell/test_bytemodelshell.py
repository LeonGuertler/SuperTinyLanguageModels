import pytest
import torch
from unittest.mock import Mock, patch
from models.experimental.byte_level.byte_model_shell import ByteModelShell  # Replace with the actual import path

@pytest.fixture
def mock_embedding_model():
    embedding_model = Mock() # Mock the embedding model

    ## Mock the forward method
    # Originally, it takes a tensor of token ids and returns the embeddings.
    # We can mock it to return embeddings for a given tensor of token embeddings.
    embedding_model.return_value = torch.rand(1, 3, 512)  # Assuming embedding size is 512 and batch size is 1 and sequence length is 3

    return embedding_model

@pytest.fixture
def mock_core_model():
    core_model = Mock()

    ## Mock the forward method
    # Originally, it takes the embeddings and returns the hidden states.
    # We can mock it to return hidden states for a given tensor of embeddings.
    core_model.return_value = torch.rand(1, 3, 512)  # Assuming hidden size is 512

    return core_model

@pytest.fixture
def mock_model_head():
    model_head = Mock()

    ## Mock the forward method
    # Originally, it takes the hidden states and returns the logits into a shape of (batch_size, sequence_len, byte_context_window, byte_vocab_size)
    # We can mock it to return logits for a given tensor of hidden states.
    model_head.return_value = torch.rand(1, 3, 12, 258), None  # Assuming byte_context_window = 12 and byte_vocab_size = 258

    return model_head

@pytest.fixture
def byte_model_shell(mock_embedding_model, mock_core_model, mock_model_head):

    ## Create a model shell with the mocked components
    model = ByteModelShell(mock_embedding_model, mock_core_model, mock_model_head)
    model.device = torch.device("cpu")
    return model


def test_forward(byte_model_shell):

    ## Create a tensor input
    input_tensor = torch.randint(0, 258, (1, 3, 12)) # (batch_size, sequence_len, byte_context_window)

    ## Call the forward method
    result, _ = byte_model_shell.forward(input_tensor) 
    
    ## Check if the methods are called correctly
    byte_model_shell.embedding_model.assert_called_once_with(input_tensor) # Check if the embedding model is called with the input tensor
    byte_model_shell.core_model.assert_called_once() # Check if the core model is called
    byte_model_shell.model_head.call_count > 0 # Check if the model head is called more than once
    assert isinstance(result, torch.Tensor) # Check if the result is a tensor
