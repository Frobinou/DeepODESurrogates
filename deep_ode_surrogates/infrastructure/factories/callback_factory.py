from deep_ode_surrogates.infrastructure.training.callbacks.checkpoint_callback import (
    CheckpointCallback,
)
from deep_ode_surrogates.infrastructure.training.callbacks.earlystopping_callback import (
    EarlyStoppingCallback,
)
from deep_ode_surrogates.infrastructure.training.callbacks.finalevaluation_callback import (
    FinalEvaluationCallback,
)
from deep_ode_surrogates.infrastructure.training.callbacks.tensorboard_callback import (
    TensorBoardCallback,
)
from deep_ode_surrogates.infrastructure.training.evaluator.base import Evaluator
from deep_ode_surrogates.infrastructure.training.schemas import CallbackConfig


def build_callbacks(callback_config: CallbackConfig, evaluators: list[Evaluator]):
    callbacks = []

    if callback_config.checkpoint is not None:
        callbacks.append(CheckpointCallback(config=callback_config.checkpoint))

    if callback_config.early_stopping is not None:
        callbacks.append(
            EarlyStoppingCallback(config=callback_config.early_stopping),
        )

    for evaluator in evaluators:
        callbacks.append(FinalEvaluationCallback(evaluator))

    if callback_config.tensorboard:
        callbacks.append(TensorBoardCallback(config=callback_config.tensorboard))
    return callbacks
