import pytest
import torch
from unittest.mock import Mock, patch
from models.model_shell import ModelShell  # Replace with the actual import path

@pytest.fixture
def mock_embedding_model():
    embedding_model = Mock() # Mock the embedding model

    ## Mock the tokenize_input method
    # Originally, it takes a string and returns a list of token ids.
    # We can mock it to return a list of token ids for a given string.
    embedding_model.tokenize_input.return_value = [1, 2, 3]

    ## Mock the pad_batch method
    # Originally, it takes a list of token lists and returns a padded tensor and mask tensor.
    # We can mock it to return a padded tensor and mask tensor for a given list of token lists.
    embedding_model.pad_batch.return_value = (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))

    ## Mock the forward method
    # Originally, it takes a tensor of token ids and returns the embeddings.
    # We can mock it to return embeddings for a given tensor of token ids.
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
    # Originally, it takes the hidden states and returns the logits.
    # We can mock it to return logits for a given tensor of hidden states.
    model_head.return_value = torch.rand(1, 3, 1000), None  # Assuming vocab size is 1000

    ## Mock the inference method
    # Originally, it takes the hidden states and returns the logits.
    # We can mock it to return logits for a given tensor of hidden states.
    model_head.inference.return_value = torch.rand(1, 1000) # Assuming logits

    return model_head

@pytest.fixture
def model_shell(mock_embedding_model, mock_core_model, mock_model_head):

    ## Create a model shell with the mocked components
    model = ModelShell(mock_embedding_model, mock_core_model, mock_model_head)
    model.device = torch.device("cpu")
    return model


def test_forward(model_shell):

    ## Create a tensor input
    input_tensor = torch.tensor([[1, 2, 3]])

    ## Call the forward method
    result, _ = model_shell.forward(input_tensor)
    
    ## Check if the methods are called correctly
    model_shell.embedding_model.assert_called_once_with(input_tensor) # Check if the embedding model is called with the input tensor
    model_shell.core_model.assert_called_once() # Check if the core model is called
    model_shell.model_head.assert_called_once() # Check if the model head is called
    assert isinstance(result, torch.Tensor) # Check if the result is a tensor


def test_inference_with_string_input(model_shell):

    ## Create a string input
    input_string = "Hello, world!"

    ## Call the inference method
    logits, model_input = model_shell.inference(input_string) # Logits should be the shape of (1, 1000) and model_input should be a list of token ids
    
    ## Check if the methods are called correctly
    model_shell.embedding_model.tokenize_input.assert_called_with(input_string, truncate=True, add_eot=False) # Check if the input string is tokenized
    model_shell.embedding_model.assert_called_once() # Check if the embedding model is called
    model_shell.core_model.assert_called_once() # Check if the core model is called
    model_shell.model_head.inference.assert_called_once() # Check if the model head is called
    assert isinstance(logits, torch.Tensor) # Check if the logits are a tensor
    assert isinstance(model_input, list) # Check if the model input is a list


def test_inference_with_input_ids(model_shell):

    ## Create a tensor input
    input_ids = torch.tensor([[1, 2, 3]])

    ## Call hte inference method
    logits, model_input = model_shell.inference(input_ids) # Logits should be the shape of (1, 1000) and model_input should be a list of token ids

    ## check if the methods are called correctly
    assert not model_shell.embedding_model.tokenize_input.called # Check if the tokenization is not called
    model_shell.embedding_model.assert_called_once() # Check if the embedding model is called
    model_shell.core_model.assert_called_once() # Check if the core model is called
    model_shell.model_head.inference.assert_called_once() # Check if the model head is called
    assert isinstance(logits, torch.Tensor) # Check if the logits are a tensor
    assert torch.equal(model_input, input_ids) # Check if the model input is the same as the input tensor


def test_loglikelihood(model_shell):

    ## Create prefixes and continuations
    prefixes = ["hey"]
    continuations = ["world"]
    
    # Mock the forward method to return a proper tensor
    model_shell.forward = Mock(return_value=(torch.rand(2, 2, 1000), None))  # Assuming batch size 2, sequence length 3, vocab size 1000
    
    ## Call the loglikelihood method
    ll = model_shell.loglikelihood(prefixes, continuations)
    
    ## Check if the methods are called correctly
    assert model_shell.embedding_model.tokenize_input.call_count > 0 # Check if the tokenization is called
    assert model_shell.embedding_model.pad_batch.call_count > 0 # Check if the padding is called
    assert model_shell.forward.called # Check if the forward method is called
    assert isinstance(ll, torch.Tensor) # Check if the loglikelihood is a tensor
    assert ll.shape == torch.Size([1]) # Check if the shape of the loglikelihood tensor is correct