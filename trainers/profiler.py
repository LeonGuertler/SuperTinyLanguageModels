"""
The profiler should be an easy to use wrapper around an arbitrary
trainer class and, on a subset of the training data, track relevant 
metrics (including time and memory usage) for each aspect of the 
training pipeline (i.e. model, optimizer, scheduler, dataloader etc.)
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity


class Trainer
class TrainerProfiler:
    def __init__(self, trainer, subset_size=1000):
        self.trainer = trainer
        self.subset_size = subset_size
        self.profiler = None

    def start_profiling(self):
        self.profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                                record_shapes=True, 
                                profile_memory=True, 
                                with_stack=True)

        self.profiler.__enter__()

    def stop_profiling(self):
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace("trace.json")  # Export results to a file for visualization

    def profile_function(self, func_name, *args, **kwargs):
        with record_function(func_name):
            getattr(self.trainer, func_name)(*args, **kwargs)

    def profile_model(self):
        # Example to profile the model's forward pass
        self.profile_function("_run_step")

    # Add more methods as needed to profile different components
