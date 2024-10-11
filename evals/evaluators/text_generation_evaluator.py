import torch 
from evals.core import BaseModelWrapper
from typing import Callable, List, Dict, Any

class TextGenerationEvaluator:
    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        yield_fn: Callable,
        yield_fn_params: Optional[Dict[str, Any]] = None,
        eval_metric: Optional[str] = "LLM-PPL",
        eval_logging_path: Optional[str] = "Text Generation" 
    ):
        super().__init__()
        pass 
        # TODO 