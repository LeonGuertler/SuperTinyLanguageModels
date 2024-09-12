"""
A collection of util functions
"""
import os 
import hydra 


### Tokenizer Utils
def get_tokenizer_path(tokenizer_type, vocab_size, dataset_name):
    """
    Get the path to the tokenizer.
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ## Hot-fix because the hydra.utils.get_original_cwd() is not working 
    ## TODO: find better solution
    
    tokenizer_folder = os.path.join(
        project_root, "components", "layers", "tokenizer_models"
    )
    tokenizer_full_path = os.path.join(
        tokenizer_folder, f"{tokenizer_type}_{dataset_name}_{vocab_size}.model"
    )
    input(tokenizer_full_path)
    return tokenizer_folder, tokenizer_full_path

def check_if_tokenizer_exists(tokenizer_type, vocab_size, dataset_name):
    """
    Check if the tokenizer already exists.
    """
    _, tokenizer_path = get_tokenizer_path(tokenizer_type, vocab_size, dataset_name)
    return os.path.exists(tokenizer_path)

