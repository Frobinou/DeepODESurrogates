# infrastructure/training/training_pipeline.py

from pydantic import BaseModel

from deep_ode_surrogates.infrastructure.training.callbacks.schemas import (
    CheckpointCallbackConfig,
    EarlyStoppingCallbackConfig,
    TensorboardCallbackConfig,
)


class CallbackConfig(BaseModel):
    tensorboard: TensorboardCallbackConfig | None = TensorboardCallbackConfig()
    checkpoint: CheckpointCallbackConfig | None = CheckpointCallbackConfig()
    early_stopping: EarlyStoppingCallbackConfig | None = EarlyStoppingCallbackConfig()
