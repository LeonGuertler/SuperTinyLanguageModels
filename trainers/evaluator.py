"""Code for running samples from the evaluation benchmarks"""
import evals
from tqdm import tqdm

def intra_training_evaluation(model, benchmarks):
    """
    Evaluates the model on multiple benchmarks during training.
    
    Args:
        model: The model to evaluate.
        benchmarks List[str]: A list of benchmark names to evaluate the model on.
    """
    results_dict = {}

    # Outer progress bar for benchmarks
    with tqdm(benchmarks, desc="Evaluating benchmarks", position=0,  leave=True) as benchmark_bar:
        for benchmark in benchmark_bar:
            # Create the benchmark evaluator
            benchmark_evaluator = evals.make(benchmark)

            # Evaluating within the benchmark, tqdm already exists in the yield function
            results = benchmark_evaluator.evaluate(model=model)
            results_dict.update(results)

    return results_dict