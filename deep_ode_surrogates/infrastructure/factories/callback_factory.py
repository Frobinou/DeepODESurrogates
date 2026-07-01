from deep_ode_surrogates.infrastructure.persistence.checkpoints.checkpoint_manager import (
    CheckpointManager,
)
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


def build_callbacks(experiment_path, training_config, callback_config, evaluators, logger=None):
    callbacks = []

    if callback_config.use_tensorboard:
        callbacks.append(TensorBoardCallback(experiment_path / "tensorboard_logs"))

    if callback_config.use_checkpoint:
        manager = CheckpointManager(
            save_dir=experiment_path / "save",
            top_k=training_config.checkpoint_k,
            logger=logger,
        )
        callbacks.append(CheckpointCallback(manager))

    if callback_config.use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                patience=callback_config.early_stopping_patience,
            )
        )

    for evaluator in evaluators:
        callbacks.append(FinalEvaluationCallback(evaluator))
    return callbacks
