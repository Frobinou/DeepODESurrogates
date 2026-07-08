from pydantic import Field
from pydantic.dataclasses import dataclass

from deep_ode_surrogates.application.config.task import TaskConfig
from deep_ode_surrogates.domain.models import AvailablesAIModel


class TrainingConfig(TaskConfig):
    """Pure training hyperparameters."""

    epochs: int = Field(2000, gt=0)
    lr: float = Field(1e-3, gt=0.0)
    checkpoint_k: int = Field(5, gt=0)
    model_name: AvailablesAIModel = AvailablesAIModel.BASIC_PINN
    optimizer: str = "Adam"  # for now we only support Adam, but this allows to easily add more optimizers in the future
    evaluators_frequency: int = 10


@dataclass
class TrainingPipeline:
    trainer: object
    dataloader: object
    callbacks: list
    evaluators: list
