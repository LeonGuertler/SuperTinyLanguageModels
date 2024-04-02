import torch
import pandas as pd



def print_model_stats(model):
    """
    Print relevant model statistics, including the number of parameters
    with and without embeddings for a given PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())

    embeddings_params = sum(p.numel() for p in model.embedder.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())

    # check if the parameters are shared
    shared_embedding = model.embedder.embedding.weight is model.lm_head.linear.weight
    if shared_embedding:
        core_model_params = total_params - embeddings_params
        lm_head_and_embeddings_params = lm_head_params
    else:
        core_model_params = total_params - embeddings_params - lm_head_params
        lm_head_and_embeddings_params = lm_head_params + embeddings_params
    
    # Prepare the data
    data = {
        "Component": ["Total", "lm_head_and_embeddings_params", "Core Model"],
        "Parameters": [total_params, lm_head_and_embeddings_params, core_model_params]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Print the table
    print(df.to_string(index=False))



"""from datasets import load_dataset




def count_params(model):
    return {
        "all_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "all": sum(p.numel() for p in model.parameters()),
        "emb": sum(p.numel() for p in model.parameters() if "emb" in p.name),
    }


DATASET_LOADERS = {
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
}


# dataset loader functions
def load_datasets(dataset_name, shuffle=True):
    dataset = DATASET_LOADERS[dataset_name]()

    # cerate dataset split
    split_dataset = dataset["train"].train_test_split(
        test_size=0.01, seed=489, shuffle=shuffle
    )

    # return the training and validation datasets
    return split_dataset
"""