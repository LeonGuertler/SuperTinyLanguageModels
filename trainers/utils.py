"""Utilities for the trainer"""

import importlib
import inspect
import os
import pkgutil

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets

import torch.distributed as dist

import hydra

def set_seed(seed):
    """Setup the trainer"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def create_folder_structure(path_config):
    """
    Create all necessary folders for training.
    """
    if not os.path.exists(path_config["data_dir"]):
        os.makedirs(path_config["data_dir"])

    if not os.path.exists(path_config["checkpoint_dir"]):
        os.makedirs(path_config["checkpoint_dir"])

def create_stlm_data_mix():
    """
    A small custom datamix for STLM models containing:
    - simple English Wikipedia
    - Python Code (Deepmind Code Contest) - sampled for easy questions
    - technical QA style (StackExchange)
    """
    # Load simple English Wikipedia
    wiki = load_dataset("wikimedia/wikipedia", "20231101.simple")["train"]

    # Add a "text" column for simple English Wikipedia
    wiki = wiki.map(lambda x: {"text": x["text"]})

    # Load Python code from DeepMind Code Contests
    code_dataset = load_dataset("jtatman/python-code-dataset-500k")["train"]
    code_dataset = code_dataset.map(lambda x: {"text": f"Instruction: {x['instruction']}\nOutput: {x['output']}"})


    # Load technical QA style data from StackExchange
    openhermes = load_dataset("teknium/OpenHermes-2.5")["train"]

    # Transform to have a "text" column with both question and answers
    openhermes = openhermes.map(lambda x: {"text": f"Question: {x['conversations'][0]['value']}\nAnswers: {x['conversations'][1]['value']}"})

    # Add tiny stories
    tiny_stories = load_dataset("roneneldan/TinyStories")["train"]
    tiny_stories = tiny_stories.map(lambda x: {"text": f"Title: {x['title']}\nStory: {x['story']}"})


    # Calculate and print the distribution of string lengths
    def calculate_length_distribution(dataset):
        lengths = [len(item["text"]) for item in dataset]
        return sum(lengths), lengths

    wiki_length, wiki_lengths = calculate_length_distribution(wiki)
    python3_code_length, python3_code_lengths = calculate_length_distribution(code_dataset)
    openhermes_length, openhermes_lengths = calculate_length_distribution(openhermes)
    tiny_stories_length, tiny_stories_lengths = calculate_length_distribution(tiny_stories)

    total_length = wiki_length + python3_code_length + openhermes_length + tiny_stories_length

    print(f"Wiki Text Length: {wiki_length} ({wiki_length/total_length*100:.2f}%)")
    print(f"Python Code Text Length: {python3_code_length} ({python3_code_length/total_length*100:.2f}%)")
    print(f"openhermes Text Length: {openhermes_length} ({openhermes_length/total_length*100:.2f}%)")

    # Concatenate datasets
    combined_dataset = concatenate_datasets([wiki, code_dataset, openhermes, tiny_stories])

    combined_dataset = DatasetDict({
        "train": combined_dataset,
    })

    return combined_dataset



DATASET_DICT = {
    "debug": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "babylm_100m": lambda: load_dataset("Sree1994/babylm_100M"), # https://babylm.github.io/
    "tinystories": lambda: load_dataset("roneneldan/TinyStories"), # https://huggingface.co/datasets/roneneldan/TinyStories
    "stlm": create_stlm_data_mix,
    "openhermes-2.5": lambda: load_dataset("teknium/OpenHermes-2.5"),
    "openwebtext": lambda: load_dataset("Skylion007/openwebtext")
}


def load_data(dataset_name, shuffle=True):
    """Load the data"""
    assert dataset_name in DATASET_DICT, f"Dataset {dataset_name} not found!"
    dataset = DATASET_DICT[dataset_name]()

    # create dataset split
    split_dataset = dataset["train"].train_test_split(
        test_size=0.01, seed=489, shuffle=shuffle
    )

    # rename test split to val
    split_dataset["val"] = split_dataset.pop("test")

    if dataset_name == "debug":
        split_dataset["train"] = split_dataset["train"].select(range(2048))

    # return the training and validation datasets
    return split_dataset


def get_classes_from_module(module_name):
    """
    Get a list of classes defined in a module or package.

    Args:
        module_name (str): The name of the module or package.

    Returns:
        list: A list of classes defined in the module or package.
    """
    module = importlib.import_module(module_name)
    classes = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if inspect.getmodule(obj) == module:
            classes.append(obj)

    return classes


def get_classes_from_package(package_name):
    """
    Get a list of classes defined in a package and its subpackages.

    Args:
        package_name (str): The name of the package.

    Returns:
        list: A list of classes defined in the package and its subpackages.
    """
    package = importlib.import_module(package_name)
    classes = get_classes_from_module(package_name)

    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        classes.extend(get_classes_from_module(module_name))

    return classes


def register_backward_hooks(tensor, module_name):
    """Registers hooks to profile the backward pass of a tensor."""
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:

        def backward_hook(grad):
            with torch.autograd.profiler.record_function(f"{module_name}.backward"):
                return grad

        tensor.register_hook(backward_hook)


def profilize(model, classes=None):
    """Recursively add hooks to the model for recording PyTorch profiler traces with module names"""
    if classes is None:
        classes = get_classes_from_package("models")
        classes += get_classes_from_package("models.components.layers")
        print(f"Found classes for profiling: {classes}")

    for module in model.children():
        if isinstance(module, torch.nn.Module):
            profilize(module, classes=classes)
        if isinstance(module, torch.nn.ModuleDict):
            for sub_module in module.values():
                profilize(sub_module, classes=classes)
        if isinstance(module, torch.nn.ModuleList):
            for sub_module in module:
                profilize(sub_module, classes=classes)

    if (
        hasattr(model, "forward")
        and any(isinstance(model, cls) for cls in classes)
        and not hasattr(model, "old_forward")
    ):
        model.old_forward = model.forward
        print(f"added forward profiling wrapper for {model.__class__.__name__}")

        def forward_wrapper(*args, **kwargs):
            nested_module_name = model.__class__.__name__
            with torch.autograd.profiler.record_function(
                f"{nested_module_name}.forward"
            ):
                outputs = model.old_forward(*args, **kwargs)
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    register_backward_hooks(output, nested_module_name)
            else:
                register_backward_hooks(outputs, nested_module_name)
            return outputs

        model.forward = forward_wrapper

def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def aggregate_value(value, device = torch.device("cuda")): 
    """
    Since using DDP, calculation of metrics happen across all GPUs. 
    This function aggregate the loss across all GPUs. 
    """
    if not is_dist_avail_and_initialized():
        return value
    all_loss = torch.tensor([value], device=device)
    dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)
    return all_loss.item() / dist.get_world_size()

def init_print_override():
    '''
    Overriding the print function is useful when running DDP. 
    This way, only rank 0 prints to the console.
    '''
    import builtins as __builtin__
    
    original_print = __builtin__.print

    def print(*args, **kwargs):
        if os.getenv('GLOBAL_RANK') == '0':
            original_print(*args, **kwargs)

    __builtin__.print = print

    return original_print

def restore_print_override(original_print):
    '''
    Restore the original print function.
    '''
    import builtins as __builtin__
    __builtin__.print = original_print

def init_kd_cfg(cfg):
    """
    Initialize the knowledge distillation config.
    """
    
    temperature = cfg.teachermodel.temperature
    embedding_loss_weight = cfg.teachermodel.embedding_loss_weight
    attn_loss_weight = cfg.teachermodel.attn_loss_weight
    hs_loss_weight = cfg.teachermodel.hs_loss_weight
    soft_targets_loss_weight = cfg.teachermodel.soft_targets_loss_weight
    label_loss_weight = cfg.teachermodel.label_loss_weight

    kd_cfg = {
        "temperature": temperature,
        "embedding_loss_weight": embedding_loss_weight,
        "attn_loss_weight": attn_loss_weight,
        "hs_loss_weight": hs_loss_weight,
        "soft_targets_loss_weight": soft_targets_loss_weight,
        "label_loss_weight": label_loss_weight
    }

    return kd_cfg

def get_qk_scores(model, inputs):
    '''
    Get the q and k projections from the model.
    '''
    raw_attentions = []
    hooks = []

    ## create a hook to capture the raw attention logits
    def attention_hook(module, input, output):
        # Capture the raw attention logits (QK^T)
        qk = output
        raw_attentions.append(qk.detach())

    ## register the hook to the relevant modules of the model
    for name, module in model.named_modules():
        if 'q_proj' in name or 'k_proj' in name:
            hook = module.register_forward_hook(attention_hook)
            hooks.append(hook)

    ## forward pass
    with torch.no_grad():
        outputs = model(inputs)

    ## remove the hooks
    for hook in hooks:
        hook.remove()

    return raw_attentions


def calculate_prenormalized_attention(q, k, teacher_model = None):
    '''
    Calculate the pre-normalized attention scores.
    '''
    
    ## if teacher model is provided, we need to adjust the q and k projections
    if teacher_model is not None:
        config = teacher_model.core_model.model.config

        B, S = q.size()[:2]
        H = config.hidden_size
        nH = config.num_attention_heads
        nKV = config.num_key_value_heads

        q = q.view(B, S, nH, H // nH) # (B, S, nH, H//nH)
        k = k.view(B, S, nKV, H // nH) # (B, S, nKV, H//nH)

        q = q.transpose(1,2) # (B, nH, S, H//nH)
        k = k.transpose(1,2) # (B, nKV, S, H//nH)

        k = k.repeat_interleave(nH // nKV, dim=1) # (B, nKV, S, H//nH) -> (B, nH, S, H//nH)

    return torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)


def get_prenormalized_attention_list(raw_attentions, teacher_model = None):
    '''
    Get the attention matrix for each layer
    '''
    ## get the odd indexzes of raw attentions
    q_projs = [raw_attentions[i] for i in range(0, len(raw_attentions), 2)]
    k_projs = [raw_attentions[i] for i in range(1, len(raw_attentions), 2)]

    attention_matrices = [calculate_prenormalized_attention(q_proj, k_proj, teacher_model=teacher_model) for q_proj, k_proj in zip(q_projs, k_projs)]

    return attention_matrices

def project_student_to_teacher_hs(projection, student_hiddenstate):
    '''
    Project student hidden states to match teacher dimensions.
    '''

    from_dim = student_hiddenstate.size(-1)
    to_dim = projection.projection_hs.out_features

    if from_dim != to_dim:
        student_hiddenstate = projection.projection_hs(student_hiddenstate)

    return student_hiddenstate

def project_student_to_teacher_emb(projection, student_emb):
    '''
    Project student hidden states to match teacher dimensions.
    '''

    from_dim = student_emb.size(-1)
    to_dim = projection.projection_hs.out_features

    if from_dim != to_dim:
        student_emb = projection.projection_emb(student_emb)

    return student_emb