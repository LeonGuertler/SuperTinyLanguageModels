import pydantic

from models import model_shell
from trainers.datasets import DatasetConfig
from trainers.evaluation import EvaluationConfig
from trainers.loss_fn import LossConfig
from trainers.optimizers import OptimizerConfig
from trainers.samplers import SamplerConfig
from trainers.schedulers import DropoutSchedulerConfig, LRSchedulerConfig


class TrainerConfig(pydantic.BaseModel):
    """Base Trainer Configuration"""

    trainer_type: str
    dataset: str = "openwebtext"
    batch_size: int = 24
    gradient_accumulation_steps: int = 20
    max_iters: int = 30000
    log_interval: int = 10
    lr_decay_iters: int = 30000
    checkpoint_interval: int = 5000
    run_profiler: bool = False


class TrainConfig(pydantic.BaseModel):
    model: model_shell.ModelShellConfig
    training: TrainerConfig
    eval: EvaluationConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    dropout_scheduler: DropoutSchedulerConfig
    dataset: DatasetConfig
    sampler: SamplerConfig
    loss_fn: LossConfig
