"""
The profiler should be an easy to use wrapper around an arbitrary
trainer class and, on a subset of the training data, track relevant
metrics (including time and memory usage) for each aspect of the
training pipeline (i.e. model, optimizer, scheduler, dataloader etc.)
"""

from torch.profiler import ProfilerActivity, profile, record_function


class TrainerProfiler:
    """Profiler for the trainer class"""

    def __init__(self, trainer, subset_size=1000):
        self.trainer = trainer
        self.subset_size = subset_size
        self.profiler = None

    def start_profiling(self):
        """Start profiling the trainer class"""
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        self.profiler.start()

    def stop_profiling(self):
        """Stop profiling the trainer class"""
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(
            "trace.json"
        )  # Export results to a file for visualization

    def profile_function(self, func_name, *args, **kwargs):
        """Profile a specific function in the trainer class"""
        with record_function(func_name):
            getattr(self.trainer, func_name)(*args, **kwargs)

    def profile_model(self):
        """Profile the model's forward pass"""
        # Example to profile the model's forward pass
        self.profile_function("_run_step")

    # Add more methods as needed to profile different components
