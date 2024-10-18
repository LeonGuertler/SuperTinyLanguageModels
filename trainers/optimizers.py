import torch

# List of official PyTorch optimizers
OFFICIAL_OPTIMIZERS = [
    'SGD', 'Adam', 'AdamW', 'Adagrad', 'Adadelta',
    'RMSprop', 'SparseAdam', 'Adamax', 'ASGD',
    'LBFGS',
]

# List of optimizers from the pytorch_optimizer library
PYTORCH_OPTIMIZER_OPTIMIZERS = [
    'Lookahead', 'Ranger', 'Ranger21', 'NovoGrad',
    'DiffGrad', 'SGDP', 'Yogi', 'Lion', 'Shampoo',
    'SOAP'
]

def build_additional_optimizers():
    """
    Attempts to import the pytorch_optimizer library.
    
    Returns:
        module: The imported pytorch_optimizer module.
    
    Raises:
        ImportError: If pytorch_optimizer is not installed.
    """
    try:
        import pytorch_optimizer as optim
        return optim
    except ImportError as e:
        raise ImportError(
            "The 'pytorch_optimizer' library is not installed. "
            "Please install it using 'pip install pytorch-optimizer' "
            "to use additional optimizers."
        ) from e

def build_optimizer(optimizer_name, model, optimizer_params):
    """
    Builds and returns an optimizer based on the provided name.
    
    Args:
        optimizer_name (str): The name of the optimizer to build.
        model (torch.nn.Module): The model whose parameters the optimizer will update.
        optimizer_params (dict): A dictionary of parameters for the optimizer.
    
    Returns:
        torch.optim.Optimizer or pytorch_optimizer.Optimizer: The instantiated optimizer.
    
    Raises:
        ValueError: If the optimizer name is not recognized.
        ImportError: If pytorch_optimizer is required but not installed.
    """
    optimizer_name = optimizer_name.lower()
    
    # Create a mapping for case-insensitive comparison
    official_optimizers_map = {name.lower(): name for name in OFFICIAL_OPTIMIZERS}
    pytorch_optimizer_map = {name.lower(): name for name in PYTORCH_OPTIMIZER_OPTIMIZERS}
    
    if optimizer_name in official_optimizers_map:
        # Get the correct case-sensitive optimizer name
        official_name = official_optimizers_map[optimizer_name]
        optimizer_class = getattr(torch.optim, official_name)
        return optimizer_class(model.parameters(), **optimizer_params)
    
    elif optimizer_name in pytorch_optimizer_map:
        # Import pytorch_optimizer library
        optim = build_additional_optimizers()
        # Get the correct case-sensitive optimizer name
        additional_name = pytorch_optimizer_map[optimizer_name]
        input(optim.OPTIMIZER_LIST)
        optimizer_class = getattr(optim, additional_name)
        return optimizer_class(model.parameters(), **optimizer_params)
    
    else:
        available_official = ', '.join(OFFICIAL_OPTIMIZERS)
        available_additional = ', '.join(PYTORCH_OPTIMIZER_OPTIMIZERS)
        raise ValueError(
            f"Optimizer '{optimizer_name}' is not recognized.\n"
            f"Available official PyTorch optimizers: {available_official}\n"
            f"Available pytorch_optimizer optimizers: {available_additional}"
        )