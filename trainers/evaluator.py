"""Code for running samples from the evaluation benchmarks"""

from evals import (
    MCQEvaluator,
    TextModelingEvaluator,
    TextGenerationEvaluator
)


def train_eval_mcq(model, num_samples, benchmark_list):
    """ Create and run the MCQ evaluator """
    # wrap so failure doesn't crash the full run
    try:
        # load the MCQ evaluator
        evaluator = MCQEvaluator(
            model=model,
            num_samples=num_samples,
            benchmark_list=benchmark_list,
        )
        # run the evaluator
        return evaluator.evaluate()
    except Exception as exc:
        print(f"The MCQ evaluator failed vai: {exc}")
        return {}



def train_eval_text_modeling(model, topic_list):
    """ Test the model """
    # wrap so failure doesn't crash the full run
    try:
        # load the Text Modeling evaluator
        evaluator = TextModelingEvaluator(
            model=model,
            topic_list=topic_list,
        )
        # run the evaluator
        return evaluator.evaluate()
    except Exception as exc:
        print(f"The MCQ evaluator failed vai: {exc}")
        return {}


def train_eval_text_generation(model):
    """ Test the model stext generation capability """
    # wrap so failure doesn't crash the full run
    try:
        # load the Text Generation evaluator
        evaluator = TextGenerationEvaluator(
            model=model
        )

        # run the evaluator
        return evaluator.evaluate()
    except Exception as exc:
        print(f"The MCQ evaluator failed vai: {exc}")
        return {}, ""

