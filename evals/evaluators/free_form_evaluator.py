from evals.benchmarks.yield_functions import * 
from models.generators import BaseGenerator

from evals.core import BaseEvaluator, BaseModelWrapper
from typing import Optional, Callable, Dict, Any

import sympy

class FreeFormEvaluator(BaseEvaluator):
    """ Evaluator for free form questions. """

    def __init__(
        self, 
        yield_fn: Callable,
        answer_extraction_function: Callable,
        model_wrapper: BaseModelWrapper,
        model_generator, #: Optional[BaseGenerator] = None, # if no generated is provided, the model must be wrapped outside
        generator_params: Optional[Dict[str, Any]] = None, 
        yield_fn_params: Optional[Dict[str, Any]] = None,
        eval_logging_path: Optional[str] = "FreeForm"
    ):
        super().__init__()
        """ TODO """
        self.eval_logging_path = eval_logging_path
        self.yield_fn = yield_fn(**yield_fn_params)
        self.generator_params = generator_params
        self.extract_answer = answer_extraction_function
        self.model_wrapper = model_wrapper 
        self.model_generator = model_generator

    def _compare_math_answers(self, true_answer: str, model_answer: str) -> bool:
        """
        Compares two mathematical expressions for equivalence.
        """
        try:
            true_expr = sympy.sympify(true_answer)
            model_expr = sympy.sympify(model_answer)
            return sympy.simplify(true_expr - model_expr) == 0
        except (sympy.SympifyError, TypeError):
            # Fallback to string comparison
            return true_answer.strip() == model_answer.strip()


    def evaluate(self, model):
        """ TODO """
        # wrap the model
        model = self.model_wrapper(
            model=model,
            model_generator=self.model_generator,
            generator_params=self.generator_params
        )
        total, correct = 0, 0

        for question, answer in self.yield_fn:
            # generate model's answer
            generated_answer = model(question)


            # extract final answers
            true_answer = self.extract_answer(answer)
            model_answer = self.extract_answer(generated_answer)

            # comopare answers (numerical comparison)
            if self._compare_math_answers(
                true_answer=true_answer, 
                model_answer=model_answer
            ):
                correct += 1
            total += 1


        accuracy = correct / total if total > 0 else 0
        return {
            f"{self.eval_logging_path}/{self.env_id} (Acc.)": accuracy
        }    

