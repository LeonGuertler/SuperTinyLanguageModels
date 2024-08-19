"""
A collection of different datasamplers.
"""

from typing import Iterator

import pydantic
import torch
import torch.utils.data
from pydantic import PositiveInt


class SamplerConfig(pydantic.BaseModel):
    """Config for building the sampler
    The data source should be"""

    data_source: torch.utils.data.Dataset
    batch_size: PositiveInt


class BaseSampler(torch.utils.data.Sampler[int]):
    """
    Samples elements randomly with replacement on-demand.

    Args:
        data_source (Dataset): dataset to sample from.
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (torch.Generator): Generator used in sampling.
    """

    def __init__(self, data_source, batch_size) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.generator = torch.Generator()
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        """
        Get a batch worth of random indicies
        """
        # Generate random indices each time __iter__ is called
        return iter(
            torch.randint(
                high=self.num_samples,
                size=(self.batch_size,),
                dtype=torch.int64,
                generator=self.generator,
            ).tolist()
        )

    def __len__(self) -> int:
        return self.num_samples
