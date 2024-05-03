"""A profiler that trains on a subset of the data and keeps
track of relevant statistics (like memory and time usage of
each module)."""

import time

from torch.profiler import ProfilerActivity, profile

from trainers.base_trainer import BaseTrainer


class TimeWrapper:
    """Wrapper to track the time taken by a function."""

    def __init__(self, func):
        self.func = func
        self.time_tracker = []

    def __call__(self, *args, **kwargs):
        t0 = time.time()
        out = self.func(*args, **kwargs)
        t1 = time.time()
        self.time_tracker.append(t1 - t0)
        return out

    def get_time(self):
        """Get the time taken by the function."""
        return self.time_tracker


class TimeWrapperModel:
    """Wrapper to track the time taken by the model forward pass."""

    def __init__(self, model):
        self.model = model
        self.time_tracker_train = []
        self.time_tracker_eval = []
        self.layer_time_memory_stats = []

    def __call__(self, *args, **kwargs):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            t0 = time.time()
            out = self.model(*args, **kwargs)
            t1 = time.time()

        if self.model.training:
            self.time_tracker_train.append(t1 - t0)
        else:
            self.time_tracker_eval.append(t1 - t0)

        self._parse_profiler(prof)

        return out

    def _parse_profiler(self, profiler):
        # Parse the profiler output to store layer-wise time and memory usage
        for event in profiler.key_averages(group_by_input_shape=True):
            self.layer_time_memory_stats.append(
                {
                    "layer": event.key,
                    "cpu_time_total": event.cpu_time_total,
                    "cuda_time_total": event.cuda_time_total,
                    "cpu_memory_usage": event.cpu_memory_usage,
                    "cuda_memory_usage": event.cuda_memory_usage,
                }
            )

    def get_layer_time_memory_stats(self):
        """Get the layer-wise time and memory usage statistics."""
        return self.layer_time_memory_stats

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()

    def train(self):
        """Set the model to training mode."""
        self.model.train()

    def get_time_train(self):
        """Get the time taken by the model in training mode."""
        return self.time_tracker_train

    def get_time_eval(self):
        """Get the time taken by the model in evaluation mode."""
        return self.time_tracker_eval

    def state_dict(self):
        """Get the state dict of the model."""
        return self.model.state_dict()

    def parameters(self):
        """Get the parameters of the model."""
        return self.model.parameters()


class BaseProfiler(BaseTrainer):
    """Profiler for the trainer class"""

    def pretty_print_stats(self, time_dict):
        """Print the statistics in a pretty format."""
        # Header for the table
        print(
            f"{'Component':<25} {'Calls':<10} {'Total Time (s)':<15} {'Average Time (ms)':<20}"
        )
        print("-" * 70)

        # Loop through each component and print its stats
        for component, times in time_dict.items():
            total_time = sum(times)
            avg_time = (
                total_time / len(times)
            ) * 1000  # Converting to milliseconds for readability
            print(
                f"{component:<25} {len(times):<10} {total_time:<15.3f} {avg_time:<20.3f}"
            )

    def train(self, *_):
        # first wrap all relevant functions

        # wrap the estimate_loss function
        self.estimate_performance = TimeWrapper(self.estimate_performance)

        # wrap the _run_step function
        self._run_step = TimeWrapper(self._run_step)

        # wrap the save_model function
        self._save_model = TimeWrapper(self._save_model)

        # wrap the model forward pass
        self.model = TimeWrapperModel(self.model)

        # wrap the optimizer step
        self.optimizer.step = TimeWrapper(self.optimizer.step)

        # wrap the scheduler step
        self.lr_scheduler.step = TimeWrapper(self.lr_scheduler.step)

        # wrap the dropout scheduler step
        self.dropout_scheduler.step = TimeWrapper(self.dropout_scheduler.step)

        # wrap the get_batch function
        self.dataloader.get_batch = TimeWrapper(self.dataloader.get_batch)

        # Start profiling
        # run training loop
        self.run_training_loop()

        # get the time for each function
        time_dict = {
            "estimate_loss": self.estimate_performance.get_time(),
            "_run_step": self._run_step.get_time(),
            "_save_model": self._save_model.get_time(),
            "model_train_pass": self.model.get_time_train(),
            "model_eval_pass": self.model.get_time_eval(),
            "optimizer.step": self.optimizer.step.get_time(),
            "scheduler.step": self.lr_scheduler.step.get_time(),
            "dropout_scheduler.step": self.dropout_scheduler.step.get_time(),
            "dataloader.get_batch": self.dataloader.get_batch.get_time(),
        }

        # input(time_dict)
        self.pretty_print_stats(time_dict)

        def structure_layer_stats(layer_stats):
            """
            Organize layer statistics into a hierarchical structure.
            """
            structured_data = {}
            for stat in layer_stats:
                path = stat["layer"].split("/")
                current_level = structured_data
                for part in path:
                    if part not in current_level:
                        current_level[part] = {"_stats": [], "children": {}}
                    current_level = current_level[part]["children"]
                current_level["_stats"].append(stat)
            return structured_data

        def print_layer_stats(structured_data, indent=0):
            """
            Recursively print the structured layer statistics with indentation.
            """
            for layer, data in structured_data.items():
                if layer == "_stats":
                    continue  # Skip printing the stats directly;
                    # they are printed when accessing their parent
                avg_cpu_time = sum(d["cpu_time_total"] for d in data["_stats"]) / len(
                    data["_stats"]
                )
                avg_cuda_time = sum(d["cuda_time_total"] for d in data["_stats"]) / len(
                    data["_stats"]
                )
                avg_cpu_memory = sum(
                    d["cpu_memory_usage"] for d in data["_stats"]
                ) / len(data["_stats"])
                avg_cuda_memory = sum(
                    d["cuda_memory_usage"] for d in data["_stats"]
                ) / len(data["_stats"])
                print(
                    f"{' ' * indent}{layer}: CPU Time: {avg_cpu_time:.3f}ms,"
                    f" CUDA Time: {avg_cuda_time:.3f}ms"
                    f", CPU Memory: {avg_cpu_memory}B, CUDA Memory: {avg_cuda_memory}B"
                )
                print_layer_stats(data["children"], indent + 4)

        # Example usage:
        layer_stats = (
            self.model.get_layer_time_memory_stats()
        )  # Assuming this returns your collected stats
        structured_stats = structure_layer_stats(layer_stats)
        print_layer_stats(structured_stats)
