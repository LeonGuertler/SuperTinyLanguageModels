import re
import importlib
from typing import Any, Callable, Dict, Tuple, Optional
from dataclasses import dataclass, field

# Global environment registry
BENCHMARK_REGISTRY: Dict[str, 'BenchmarkSpec'] = {}

@dataclass
class BenchmarkSpec:
    """A specification for creating environments."""
    id: str
    entry_point: Callable
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def make(self, **kwargs) -> Any:
        """Create an benchmark instance."""
        all_kwargs = {**self.kwargs, **kwargs}
        return self.entry_point(**all_kwargs)



def register(id: str, entry_point: Callable, **kwargs: Any):
    """Register an Benchmark with a given ID."""
    if id in BENCHMARK_REGISTRY:
        raise ValueError(f"Benchmark {id} already registered.")
    BENCHMARK_REGISTRY[id] = BenchmarkSpec(id=id, entry_point=entry_point, kwargs=kwargs)

def pprint_registry():
    """Pretty print the current registry of benchmarks."""
    if not BENCHMARK_REGISTRY:
        print("No Benchmarks registered.")
    else:
        print("Benchmark Benchmarks:")
        for env_id, env_spec in BENCHMARK_REGISTRY.items():
            print(f"  - {env_id}: {env_spec.entry_point}")

def pprint_registry_detailed():
    """Pretty print the registry with additional details like kwargs."""
    if not ENV_REGISTRY:
        print("No benchmarks registered.")
    else:
        print("Detailed Registered Benchmarks:")
        for env_id, env_spec in BENCHMARK_REGISTRY.items():
            print(f"  - {env_id}:")
            print(f"      Entry Point: {env_spec.entry_point}")
            print(f"      Kwargs: {env_spec.kwargs}")


def check_env_exists(env_id: str):
    """Check if an benchmark exists in the registry."""
    if env_id not in BENCHMARK_REGISTRY:
        raise ValueError(f"Benchmark {env_id} is not registered.")
    else:
        print(f"Benchmark {env_id} is registered.")


def make(env_id: str, **kwargs) -> Any:
    """Create an benchmark instance using the registered ID."""
    if env_id not in BENCHMARK_REGISTRY:
        raise ValueError(f"Benchmark {env_id} not found in registry.")
    
    env_spec = BENCHMARK_REGISTRY[env_id]
    
    # Resolve the entry point if it's a string
    if isinstance(env_spec.entry_point, str):
        module_name, class_name = env_spec.entry_point.split(":")
        module = importlib.import_module(module_name)
        env_class = getattr(module, class_name)
    else:
        env_class = env_spec.entry_point
    
    # Pass additional keyword arguments
    env = env_class(**{**env_spec.kwargs, **kwargs})
    # set id
    env.set_env_id(env_id=env_id)
    return env



# from dataclasses import dataclass, field
# from typing import Any, Callable, Dict, Optional, Type

# # Global benchmark registry
# BENCHMARK_REGISTRY: Dict[str, 'BenchmarkSpec'] = {}





# @dataclass
# class BenchmarkSpec:
#     """A specification for benchmarks."""
#     name: str
#     evaluator_class: Type
#     load_fn: Optional[Callable] = None
#     kwargs: Dict[str, Any] = field(default_factory=dict)

#     def make_evaluator(self, model, **kwargs) -> Any:
#         """Create an evaluator instance."""
#         all_kwargs = {**self.kwargs, **kwargs}
#         return self.evaluator_class(model, benchmark_spec=self, **all_kwargs)

# def register(
#     name: str,
#     evaluator_class: Type,
#     load_fn: Optional[Callable] = None,
#     **kwargs: Any
# ):
#     """Register a benchmark with a given name."""
#     if name in BENCHMARK_REGISTRY:
#         raise ValueError(f"Benchmark {name} already registered.")
#     BENCHMARK_REGISTRY[name] = BenchmarkSpec(
#         name=name,
#         evaluator_class=evaluator_class,
#         load_fn=load_fn,
#         kwargs=kwargs
#     )

# def get_benchmark_spec(name: str) -> BenchmarkSpec:
#     """Retrieve a benchmark specification."""
#     if name not in BENCHMARK_REGISTRY:
#         raise ValueError(f"Benchmark {name} not found in registry.")
#     return BENCHMARK_REGISTRY[name]

# def list_registered_benchmarks():
#     """List all registered benchmarks."""
#     return list(BENCHMARK_REGISTRY.keys())

# # # Register benchmarks

# # # MCQ Benchmarks
# # register_benchmark(
# #     name='arc_easy',
# #     evaluator_class=MCQEvaluator,
# #     load_fn=load_arc_easy,
# #     num_samples=100  # Default parameters
# # )

# # register_benchmark(
# #     name='hellaswag',
# #     evaluator_class=MCQEvaluator,
# #     load_fn=load_hellaswag,
# #     num_samples=100
# # )

# # # Math Word Problem Benchmarks
# # register_benchmark(
# #     name='gsm8k',
# #     evaluator_class=MathWordProblemEvaluator,
# #     load_fn=load_gsm8k,
# #     num_samples=100
# # )

# # register_benchmark(
# #     name='aqua_rat',
# #     evaluator_class=MathWordProblemEvaluator,
# #     load_fn=load_aqua_rat,
# #     num_samples=100
# # )

# # # Text Generation Benchmark
# # register_benchmark(
# #     name='text_generation',
# #     evaluator_class=TextGenerationEvaluator,
# #     prompts=None  # Default prompts
# # )

# # # Add more benchmarks as needed...
