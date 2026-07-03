from pydantic import BaseModel, Field

from deep_ode_surrogates.application.config.data import DataConfig
from deep_ode_surrogates.application.config.evaluation import EvaluatorConfig
from deep_ode_surrogates.application.config.ode import ODESConfig
from deep_ode_surrogates.application.config.training import TrainingConfig
from deep_ode_surrogates.domain.losses import AvailablesLoss
from deep_ode_surrogates.infrastructure.training.schemas import CallbackConfig


class PhysicsWeights(BaseModel):
    """Loss weights."""

    name: AvailablesLoss = AvailablesLoss.PINN_LOSS
    lambda_ode: float = Field(1.0, ge=0.0)
    lambda_data: float = Field(1.0, ge=0.0)
    lambda_ic: float = Field(1.0, ge=0.0)


class ExperimentConfig(BaseModel):
    """Single object to dump / reload a full experiment."""

    ode: ODESConfig
    data: DataConfig
    physics_weights: PhysicsWeights = PhysicsWeights()
    training: TrainingConfig = TrainingConfig()
    callbacks: CallbackConfig = CallbackConfig()
    evaluation: EvaluatorConfig = EvaluatorConfig()
    device: str
