"""
General Model utils
"""

import pandas as pd


def print_model_stats(model):
    """
    Print relevant model statistics, including the number of parameters
    with and without embeddings for a given PyTorch model, formatted for better readability.
    """
    total_params = sum(p.numel() for p in model.parameters())

    embeddings_params = sum(p.numel() for p in model.token_embedder.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())

    # Check if the parameters are shared
    shared_embedding = model.token_embedder.weight is model.lm_head.linear.weight
    if shared_embedding:
        core_model_params = total_params - embeddings_params
        lm_head_and_embeddings_params = lm_head_params
    else:
        core_model_params = total_params - embeddings_params - lm_head_params
        lm_head_and_embeddings_params = lm_head_params + embeddings_params

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
