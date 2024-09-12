"""
General Model utils
"""

import pandas as pd
from models.model_shell import ModelShell

def analyze_shared_parameters(model1, model2):
    shared_params = 0
    total_params1 = 0
    total_params2 = 0
    
    # Create dictionaries of all parameters for each model
    params1 = {id(p): p for p in model1.parameters()}
    params2 = {id(p): p for p in model2.parameters()}
    
    # Find shared parameters
    shared_ids = set(params1.keys()) & set(params2.keys())
    
    # Count parameters
    for pid in params1:
        total_params1 += params1[pid].numel()
        if pid in shared_ids:
            shared_params += params1[pid].numel()
    
    for pid in params2:
        total_params2 += params2[pid].numel()
    
    return shared_params, (total_params1 + total_params2 - shared_params)


def print_model_stats(model: ModelShell):
    """
    Print relevant model statistics, including the number of parameters
    with and without embeddings for a given PyTorch model, formatted for better readability.
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Check if the parameters are shared
    
    _, lm_head_and_embeddings_params = analyze_shared_parameters(model.embedding_model, model.model_head)
    core_model_params = total_params - lm_head_and_embeddings_params

    # Format the numbers for better readability
    def format_number(n):
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(n)

    # Prepare the data
    data = {
        "Component": ["Total", "LM Head + Embeddings", "Core Model"],
        "Parameters": [
            format_number(total_params),
            format_number(lm_head_and_embeddings_params),
            format_number(core_model_params),
        ],
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Print the table
    print(df.to_string(index=False))

    return format_number(total_params)
