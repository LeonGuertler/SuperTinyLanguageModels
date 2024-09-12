"""Code for running samples from the evaluation benchmarks"""

from evals import (
    MCQEvaluator,
    TextModelingEvaluator,
)


def train_eval_mcq(model, num_samples, benchmark_list):
    """ Create and run the MCQ evaluator """
    # load the MCQ evaluator
    evaluator = MCQEvaluator(
        model=model,
        num_samples=num_samples,
        benchmark_list=benchmark_list,
    )
    # run the evaluator
    return evaluator.evaluate()



def train_eval_text_modeling(model, topic_list, eval_dir):
    """ Test the model """
    # load the Text Modeling evaluator
    evaluator = TextModelingEvaluator(
        model=model,
        topic_list=topic_list,
        eval_dir=eval_dir
    )
    # run the evaluator
    return evaluator.evaluate()