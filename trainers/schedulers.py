import torch
from omegaconf import DictConfig, OmegaConf



# List of official PyTorch schedulers that are step-wise
OFFICIAL_SCHEDULERS = [
    'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts', 'CyclicLR',
    'OneCycleLR', 'LambdaLR', 'LinearLR', 'MultiplicativeLR',
    'PolynomialLR',
]

# List of schedulers from the pytorch_scheduler library that are step-wise
ADDITIONAL_SCHEDULERS = [
    'WarmupScheduler', 'LinearWarmupCosineAnnealingLR',
    'CosineWithRestarts', 'PolynomialDecayLR',
    'WarmupStepLR',
]

def build_additional_schedulers():
    """
    Attempts to import the pytorch_scheduler library.

    Returns:
        module: The imported pytorch_scheduler module.

    Raises:
        ImportError: If pytorch_scheduler is not installed.
    """
    try:
        import pytorch_scheduler as sched
        return sched
    except ImportError as e:
        raise ImportError(
            "The 'pytorch_scheduler' library is not installed. "
            "Please install it using 'pip install pytorch-scheduler' "
            "to use additional schedulers."
        ) from e

def build_scheduler(scheduler_name, optimizer, scheduler_params, last_epoch=-1, verbose=False):
    """
    Builds and returns a step-wise scheduler based on the provided name.
    Supports optional warmup using SequentialLR with warmup steps.
    
    Args:
        scheduler_name (str): The name of the scheduler to build.
        optimizer (torch.optim.Optimizer): The optimizer to which the scheduler will be attached.
        scheduler_params (dict or DictConfig): Parameters for the scheduler, possibly including 'warmup_steps'.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
        verbose (bool, optional): If True, prints messages to stdout for each update. Default is False.
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler or pytorch_scheduler.Scheduler: The instantiated scheduler.
    
    Raises:
        ValueError: If the scheduler name is not recognized.
        ImportError: If pytorch_scheduler is required but not installed.
    """
    scheduler_name_lower = scheduler_name.lower()
    # Create a mapping for case-insensitive comparison
    official_schedulers_map = {name.lower(): name for name in OFFICIAL_SCHEDULERS}
    additional_schedulers_map = {name.lower(): name for name in ADDITIONAL_SCHEDULERS}
    
    # Safely retrieve 'warmup_steps' without modifying scheduler_params
    warmup_steps = scheduler_params.get('warmup_steps', None)
    
    # Create a new dictionary excluding 'warmup_steps'
    if isinstance(scheduler_params, DictConfig):
        scheduler_params_dict = OmegaConf.to_container(scheduler_params, resolve=True)
    else:
        scheduler_params_dict = dict(scheduler_params)
    
    # Remove 'warmup_steps' from the parameters passed to the scheduler constructors
    scheduler_params_dict.pop('warmup_steps', None)
    
    if warmup_steps is not None:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        # Build the warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,  # Start from 0% of the initial LR
            end_factor=1.0,    # Ramp up to 100% of the initial LR
            total_iters=warmup_steps,
            last_epoch=-1,
            verbose=verbose
        )

        # Build the main scheduler
        if scheduler_name_lower in official_schedulers_map:
            official_name = official_schedulers_map[scheduler_name_lower]
            scheduler_class = getattr(torch.optim.lr_scheduler, official_name, None)
            if scheduler_class is None:
                raise ValueError(
                    f"Scheduler '{official_name}' not found in torch.optim.lr_scheduler."
                )
            main_scheduler = scheduler_class(optimizer, last_epoch=-1, verbose=verbose, **scheduler_params_dict)

        elif scheduler_name_lower in additional_schedulers_map:
            sched = build_additional_schedulers()
            additional_name = additional_schedulers_map[scheduler_name_lower]
            scheduler_class = getattr(sched, additional_name, None)
            if scheduler_class is None:
                raise ValueError(
                    f"Scheduler '{additional_name}' not found in pytorch_scheduler."
                )
            main_scheduler = scheduler_class(optimizer, last_epoch=-1, verbose=verbose, **scheduler_params_dict)
        else:
            available_official = ', '.join(OFFICIAL_SCHEDULERS)
            available_additional = ', '.join(ADDITIONAL_SCHEDULERS)
            raise ValueError(
                f"Scheduler '{scheduler_name}' is not recognized.\n"
                f"Available official PyTorch schedulers: {available_official}\n"
                f"Available pytorch_scheduler schedulers: {available_additional}"
            )

        # Combine using SequentialLR
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        return scheduler

    else:
        # Build scheduler without warmup
        if scheduler_name_lower in official_schedulers_map:
            official_name = official_schedulers_map[scheduler_name_lower]
            scheduler_class = getattr(torch.optim.lr_scheduler, official_name, None)
            if scheduler_class is None:
                raise ValueError(
                    f"Scheduler '{official_name}' not found in torch.optim.lr_scheduler."
                )
            return scheduler_class(optimizer, last_epoch=last_epoch, verbose=verbose, **scheduler_params_dict)

        elif scheduler_name_lower in additional_schedulers_map:
            sched = build_additional_schedulers()
            additional_name = additional_schedulers_map[scheduler_name_lower]
            scheduler_class = getattr(sched, additional_name, None)
            if scheduler_class is None:
                raise ValueError(
                    f"Scheduler '{additional_name}' not found in pytorch_scheduler."
                )
            return scheduler_class(optimizer, last_epoch=last_epoch, verbose=verbose, **scheduler_params_dict)

        else:
            available_official = ', '.join(OFFICIAL_SCHEDULERS)
            available_additional = ', '.join(ADDITIONAL_SCHEDULERS)
            raise ValueError(
                f"Scheduler '{scheduler_name}' is not recognized.\n"
                f"Available official PyTorch schedulers: {available_official}\n"
                f"Available pytorch_scheduler schedulers: {available_additional}"
            )