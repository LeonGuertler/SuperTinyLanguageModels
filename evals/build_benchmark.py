# Build class for making benchmark classes

import evals.arc as arc
import evals.benchmark as benchmark
import evals.hellaswag as hellaswag
# import evals.mteb_benchmark as mteb_benchmark
import evals.mmlu as mmlu
import evals.vitaminc as vitaminc
import evals.nonsense as nonsense
import evals.winograd as winograd


def build_benchmark(benchmark_name, model) -> benchmark.Benchmark:
    """
    Given the benchmark name, build the benchmark
    """
    if benchmark_name == "arc":
        return arc.ARC(name=benchmark_name, model=model)
    if benchmark_name == "hellaswag":
        return hellaswag.HellaSwag(name=benchmark_name, model=model)
    # if benchmark_name == "mteb":
        # return mteb_benchmark.MTEBBenchmark(name=benchmark_name, model=model)
    if benchmark_name == "mmlu":
        return mmlu.MMLUBenchmark(name=benchmark_name, model=model)
    if benchmark_name == "vitaminc":
        return vitaminc.VitaminC(name=benchmark_name, model=model)
    if benchmark_name == "nonsense":
        return nonsense.Nonsense(name=benchmark_name, model=model)
    if benchmark_name == "winograd":
        return winograd.Winograd(name=benchmark_name, model=model)
    raise ValueError(f"Unknown benchmark name: {benchmark_name}")
