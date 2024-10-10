import torch
from evals.core import BaseModelWrapper

from typing import List


def batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class LoglikelihoodMCQModelWrapper(BaseModelWrapper):
    """ TODO """
    def __init__(self, model):
        """ TODO """
        super().__init__()
        self.model = model
        self.device = model.device

    @torch.no_grad()
    def __call__(
        self,
        prefixes: List[str], 
        continuations: List[str]
    ) -> List[float]:
        """ Compute the loglikelihood of given inputs """
        results = []
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                for prefix_batch, cont_batch in zip(
                    batch(prefixes, 32), batch(continuations, 32) # TODO (legacy) should not be hardcoded
                ):
                    ll = self.model_shell.loglikelihood(prefix_batch, cont_batch)
                    results.extend(ll.cpu().numpy())
        return results


