from evals.benchmarks.yield_functions import * 

from evals.core import BaseEvaluator, BaseModelWrapper
from typing import Optional, Callable, Dict, Any


class FreeFormEvaluator(BaseEvaluator):
    """ Evaluator for free form 