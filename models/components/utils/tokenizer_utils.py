"""
A collection of util functions
"""
import os 
import hydra 


### Tokenizer Utils
def get_tokenizer_path(tokenizer_type, vocab_size, dataset_name, simplify, num_reserved_tokens):
    """
    Get the path to the tokenizer.
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ## Hot-fix because the hydra.utils.get_original_cwd() is not working 
    ## TODO: find better solution
    
    tokenizer_folder = os.path.join(
        project_root, "components", "tokenizer_models"
    )

    # create folder if not exists
    if not os.path.exists(tokenizer_folder):
        os.mkdir(tokenizer_folder)

    tokenizer_full_path = os.path.join(
        tokenizer_folder, f"{tokenizer_type}_{dataset_name}_{vocab_size}_{num_reserved_tokens}.model"
    )
    if simplify:
        tokenizer_full_path = tokenizer_full_path.replace(".model", "_simplified.model")
    return tokenizer_folder, tokenizer_full_path

def check_if_tokenizer_exists(tokenizer_type, vocab_size, dataset_name, simplify, num_reserved_tokens):
    """
    Check if the tokenizer already exists.
    """
    _, tokenizer_path = get_tokenizer_path(
        tokenizer_type=tokenizer_type, 
        vocab_size=vocab_size, 
        dataset_name=dataset_name,
        simplify=simplify,
        num_reserved_tokens=num_reserved_tokens,
    )
    return os.path.exists(tokenizer_path)

