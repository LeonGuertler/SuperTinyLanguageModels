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
        "all_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "all": sum(p.numel() for p in model.parameters()),
        "emb": sum(p.numel() for p in model.parameters() if "emb" in p.name),
    }


DATASET_LOADERS = {
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
}


# dataset loader functions
def load_training_datasets(dataset_name, shuffle=True):
    """Loads datasets for pretraining language models"""
    dataset = DATASET_LOADERS[dataset_name]()

    # cerate dataset split
    split_dataset = dataset["train"].train_test_split(
        # these datasets only have a train split and must be split manually
        test_size=0.01,
        seed=489,
        shuffle=shuffle,
    )

    # return the training and validation datasets
    return split_dataset["train"]
