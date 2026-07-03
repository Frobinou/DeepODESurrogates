# infrastructure/training/training_pipeline_factory.py

from pathlib import Path

from deep_ode_surrogates.application.config.experiment import ExperimentConfig
from deep_ode_surrogates.application.config.training import TrainingPipeline
from deep_ode_surrogates.infrastructure.factories.callback_factory import build_callbacks
from deep_ode_surrogates.infrastructure.factories.evaluator_factory import build_evaluators
from deep_ode_surrogates.infrastructure.factories.trainer_factory import build_trainer
from deep_ode_surrogates.infrastructure.persistence.experiments.experiment_io import (
    save_experiment_config,
)
from deep_ode_surrogates.infrastructure.registries.dataloader_registry import dataloader_registry


def build_training_pipeline(experiment_config: ExperimentConfig, output_dir: Path):
    save_experiment_config(
        experiment_config,
        output_dir=output_dir,
    )

    dataloader = dataloader_registry.create(
        experiment_config.data.type, data_config=experiment_config.data
    )

    trainer = build_trainer(
        ode_config=experiment_config.ode,
        loss_config=experiment_config.physics_weights,
        training_config=experiment_config.training,
        device=experiment_config.device,
    )

    evaluators = build_evaluators(
        evaluation_config=experiment_config.evaluation, data_loader=dataloader
    )

    callbacks = build_callbacks(callback_config=experiment_config.callbacks, evaluators=evaluators)

    return TrainingPipeline(
        trainer=trainer,
        dataloader=dataloader,
        callbacks=callbacks,
        evaluators=evaluators,
    )
