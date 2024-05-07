"""
Load a bechmark loader, given the benchmark name.
"""

from evals.mcqs.benchmarks.arc import load_arc

EVALS_DICT = {
    "arc": load_arc,
}


def load_benchmark(benchmark_name):
    """
    Given the benchmark name, build the benchmark
    """
    return EVALS_DICT[benchmark_name]()
