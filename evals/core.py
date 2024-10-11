from typing import Dict, Union, Any

class BaseEvaluator:
    """Base class for all evaluators."""

    def __init__(self) -> None:
        # for better logging
        self.eval_metric: str = ...
        self.eval_logging_path: str = ...

    def set_env_id(self, env_id: str) -> None:
        """ TODO """
        self.env_id = env_id

    def evaluate(self, model): # -> Dict[str: Any]:
        """Each evaluator must implement its own evaluate method."""
        raise NotImplementedError("Each evaluator must implement its own evaluate method.")


class BaseModelWrapper:
    """ Base class for all model wrapper. """

    def __init__(self):
        pass 

    def __call__(self, **kwargs):
        """ pass the forward call through the model """
        raise NotImplementedError("Each ModelWrapper must implement its own call method.")