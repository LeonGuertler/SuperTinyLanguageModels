"""
Load a bechmark loader, given the benchmark name.
"""

from evals.mcqs.benchmarks.arc import load_arc
from evals.mcqs.benchmarks.winogrande import load_winogrande
from evals.mcqs.benchmarks.mmlu import load_mmlu
from evals.mcqs.benchmarks.hellaswag import load_hellaswag
from evals.mcqs.benchmarks.blimp import load_blimp

EVALS_DICT = {
    "arc": load_arc,
    "winograd": load_winogrande,
    "mmlu": load_mmlu,
    "hellaswag": load_hellaswag,
    "blimp": load_blimp,
}


def load_benchmark(benchmark_name, split):
    """
    Given the benchmark name, build the benchmark
    """
    return EVALS_DICT[benchmark_name](split=split)
