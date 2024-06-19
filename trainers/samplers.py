"""
A collection of different datasamplers.
"""
import torch 
from typing import Iterator, Optional, Sized


class BaseSampler(torch.utils.data.Sampler[int]):
    """
    Samples elements randomly with replacement on-demand.

    Args:
        data_source (Dataset): dataset to sample from.
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (torch.Generator): Generator used in sampling.
    """

    def __init__(self, data_source: Sized) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.generator = torch.Generator()

    def __iter__(self) -> Iterator[int]:
        # Generate random indices each time __iter__ is called
        return iter(torch.randint(
            high=self.num_samples, 
            size=(self.num_samples,), 
            dtype=torch.int64, 
            generator=self.generator).tolist()
        )

    def __len__(self) -> int:
        return self.num_samples

