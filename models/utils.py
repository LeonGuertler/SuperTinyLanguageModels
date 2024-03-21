import torch 
from datasets import load_dataset



def count_params(model):
    """
    Return a dict with four counts:
        - all trainable parameters
        - all parameters
        - all embedding parameters
        - all embedding and output paramters
    """
    return {
        'all_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'all': sum(p.numel() for p in model.parameters()),
        'emb': sum(p.numel() for p in model.parameters() if 'emb' in p.name),
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
        test_size=0.01, 
        seed=489, 
        shuffle=shuffle
    )

    # return the training and validation datasets
    return split_dataset["train"]